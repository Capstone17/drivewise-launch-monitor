import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

# Ensure Ultralytics can write its settings locally and skip auto-installation
os.environ.setdefault(
    "YOLO_CONFIG_DIR", os.path.join(os.path.dirname(__file__), ".yolo")
)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # pragma: no cover - fallback when tflite-runtime is unavailable
    from tensorflow.lite.python.interpreter import Interpreter

# Light Pi-friendly OpenCV tweaks
try:
    cv2.setNumThreads(2)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

MODEL_IMG_H = 128
MODEL_IMG_W = 192

ACTUAL_BALL_RADIUS = 2.38
FOCAL_LENGTH = 1755.0  # pixels

DYNAMIC_MARKER_LENGTH = 2.38
MIN_BALL_RADIUS_PX = 9  # pixels
EDGE_MARGIN_PX = 1
BALL_SCORE_THRESHOLD = 0.25
MOTION_WINDOW_SCORE_THRESHOLD = 0.1
MOTION_WINDOW_MIN_ASPECT_RATIO = 0.65
MOTION_WINDOW_FORWARD_SHIFT = 10
MAX_CENTER_JUMP_PX = 120.0

MOTION_WINDOW_DEBUG = os.environ.get("MOTION_WINDOW_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

try:
    _early_exit_env = int(os.environ.get("MOTION_WINDOW_EARLY_EXIT_MISSES", "0"))
except ValueError:
    _early_exit_env = 0
MOTION_WINDOW_EARLY_EXIT_MISSES = _early_exit_env if _early_exit_env > 0 else None
MOTION_WINDOW_MAX_SCAN_MULTIPLIER = 4
MOTION_WINDOW_MIN_SCAN_BUDGET = 1024
MOTION_WINDOW_DECODE_FAILURE_LIMIT = 64

# Load camera calibration parameters
_calib_path = os.path.join(os.path.dirname(__file__), "calibration", "camera_calib.npz")
_calib_data = np.load(_calib_path)
CAMERA_MATRIX = _calib_data["camera_matrix"]
DIST_COEFFS = _calib_data["dist_coeffs"]

FX = float(CAMERA_MATRIX[0, 0])
FY = float(CAMERA_MATRIX[1, 1])
CX = float(CAMERA_MATRIX[0, 2])
CY = float(CAMERA_MATRIX[1, 2])

MAX_MISSING_FRAMES = 12
MAX_MOTION_FRAMES = 40
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
USE_BLUR = False

# Reflective dot tracking parameters (mirroring MATLAB demo defaults)
DOT_MIN_AREA_PX = 15
DOT_MAX_AREA_PX = 400
DOT_MIN_BRIGHTNESS = 60.0
DOT_MIN_CIRCULARITY = 0.25
DOT_MAX_DETECTIONS = 20
DOT_MIN_Y_PX = 40.0
DOT_MIN_Y_FRACTION = 0.05
DOT_MIN_LEFT_COLUMN = 1
DOT_MIN_RIGHT_COLUMN = 1

CLUB_CORNER_RESIZE_WIDTH = 640
CLUB_CORNER_RESIZE_HEIGHT = 640
CLUB_CORNER_GFTT_MAX_CORNERS = 80
CLUB_CORNER_GFTT_QUALITY = 0.004
CLUB_CORNER_GFTT_MIN_DISTANCE = 6.0
CLUB_CORNER_SUBPIX_WINDOW = (5, 5)
CLUB_CORNER_SUBPIX_ZERO_ZONE = (-1, -1)
CLUB_HARRIS_BLOCK_SIZE = 5
CLUB_HARRIS_KSIZE = 3
CLUB_HARRIS_K = 0.04
CLUB_CORNER_DUPLICATE_DISTANCE_PX = 6.0
CLUB_CORNER_MIN_RESPONSE = 0.15
CLUB_CORNER_COLUMN_MIN_STRENGTH = 0.2
CLUB_CORNER_COLUMN_MAX_WIDTH_PX = 22.0
CLUB_CORNER_COLUMN_MAX_DEVIATION_PX = 5.0
CLUB_CORNER_MIN_VERTICAL_SPREAD_PX = 12.0
CLUB_CORNER_HEIGHT_SCALE = 0.02
CLUB_CORNER_MAX_TOTAL = 20
CLUB_CORNER_CONTRAST_WINDOW = 7
CLUB_CORNER_MIN_CONTRAST = 12.0
CLUB_CORNER_MAX_PER_LEFT = 7
CLUB_CORNER_MAX_PER_RIGHT = 3
CLUB_CORNER_COLUMN_GAP_REL_TOL = 0.45
CLUB_CORNER_COLUMN_GAP_ABS_TOL_PX = 14.0
CLUB_CORNER_COLUMN_GAP_MIN_EXPECTED_PX = 18.0
CLUB_CORNER_COLUMN_TOP_TOL = 4.0
CLUB_TRAJECTORY_AXIS_MIN_RESIDUAL = {
    'x': 0.35,
    'y': 0.35,
    'z': 0.55,
}
CLUB_TRAJECTORY_RESIDUAL_SIGMA = 2.5
CLUB_TRAJECTORY_NORM_SIGMA = 2.0
CLUB_TRAJECTORY_MIN_NORM = 1.0
CLUB_TRAJECTORY_MASK_ITERS = 2
CLUB_CENTER_TOP_ALIGNMENT_TOL = 5.0
CLUB_CENTER_SINGLE_COLUMN_UPSHIFT = 0.2

CLUBFACE_MAX_SPREAD_PX = 140.0
CLUBFACE_MAX_JUMP_PX = 80.0
CLUBFACE_EMA_ALPHA = 0.4
CLUBFACE_MIN_DOTS = 1
CLUBFACE_MAX_MISSES = 8
CLUBFACE_MAX_CANDIDATES = 6
CLUBFACE_MIN_CONFIDENCE = 0.35

# Pattern geometry: four markers stacked vertically on the heel side
# (2 cm separation) and two markers toward the toe, roughly aligned
# across the face. We treat the clubface center as midway between the
# heel and toe columns.
CLUBFACE_VERTICAL_SPACING_CM = 2.0
CLUBFACE_HORIZONTAL_SPACING_CM = 6.5
CLUBFACE_COLUMN_SPLIT_PX = 10.0
CLUBFACE_DEPTH_MIN_CM = 10.0
CLUBFACE_DEPTH_MAX_CM = 400.0
CLUBFACE_CENTER_OFFSET_CM = CLUBFACE_HORIZONTAL_SPACING_CM / 2.0
CLUBFACE_Z_OFFSET_CM = 30.0
CLUBFACE_VERTICAL_ALIGNMENT_PX = 40.0


@dataclass
class DotDetection:
    centroid: np.ndarray
    area: float
    brightness: float
    circularity: float
    column: str | None = None
    strength: float = 0.0


@dataclass
class RefinedTrajectory:
    coords: list[dict[str, float]]
    inlier_times: set[float]
    original_times: set[float]
    pixel_points: list[dict[str, float]]


class MotionWindowError(RuntimeError):
    """Base error for failures while establishing a ball motion window."""

    def __init__(self, message: str, *, stats: Optional[dict[str, object]] = None) -> None:
        super().__init__(message)
        self.stats = stats or {}


class MotionWindowNotFoundError(MotionWindowError):
    """Raised when the search could not find any candidate ball frames."""


class MotionWindowDegenerateError(MotionWindowError):
    """Raised when detections are present but cannot form a valid window."""


def _corner_patch_contrast(gray: np.ndarray, x: float, y: float, window: int) -> float:
    radius = max(1, int(window))
    xi = int(round(x))
    yi = int(round(y))
    x0 = max(0, xi - radius)
    y0 = max(0, yi - radius)
    x1 = min(gray.shape[1], xi + radius + 1)
    y1 = min(gray.shape[0], yi + radius + 1)
    if x1 - x0 <= 1 or y1 - y0 <= 1:
        return 0.0
    patch = gray[y0:y1, x0:x1]
    return float(patch.std())



class ClubfaceCentroidTracker:
    def __init__(
        self,
        *,
        max_jump_px: float = CLUBFACE_MAX_JUMP_PX,
        max_spread_px: float = CLUBFACE_MAX_SPREAD_PX,
        ema_alpha: float = CLUBFACE_EMA_ALPHA,
        min_dots: int = CLUBFACE_MIN_DOTS,
        min_confidence: float = CLUBFACE_MIN_CONFIDENCE,
        max_misses: int = CLUBFACE_MAX_MISSES,
        max_candidates: int = CLUBFACE_MAX_CANDIDATES,
    ) -> None:
        self.max_jump_px = float(max_jump_px)
        self.max_spread_px = float(max_spread_px)
        self.ema_alpha = float(ema_alpha)
        self.min_dots = max(1, int(min_dots))
        self.min_confidence = float(max(0.0, min(1.0, min_confidence)))
        self.max_misses = max(1, int(max_misses))
        self.max_candidates = max(1, int(max_candidates))
        self.prev_center: np.ndarray | None = None
        self.filtered_center: np.ndarray | None = None
        self.miss_streak = 0
        self.last_depth_cm: float | None = None

    def reset(self) -> None:
        self.prev_center = None
        self.filtered_center = None
        self.miss_streak = 0
        self.last_depth_cm = None

    def _register_miss(self) -> None:
        self.miss_streak += 1
        if self.miss_streak >= self.max_misses:
            self.prev_center = None
            self.filtered_center = None
            self.miss_streak = self.max_misses
            self.last_depth_cm = None

    @staticmethod
    def _weighted_center(points: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        masked_weights = weights[mask]
        total = float(masked_weights.sum())
        if total <= 1e-6:
            return None
        return (points[mask] * masked_weights[:, None]).sum(axis=0) / total

    @staticmethod
    def _cluster_spread(points: np.ndarray, mask: np.ndarray, center: np.ndarray) -> float:
        used = points[mask]
        if used.size == 0:
            return 0.0
        distances = np.linalg.norm(used - center, axis=1)
        if distances.size == 0:
            return 0.0
        return float(distances.max())

    def _confidence(self, used: int, spread: float, jump: float | None) -> float:
        count_factor = min(1.0, used / 4.0)
        if self.max_spread_px > 0.0:
            spread_factor = 1.0 - max(0.0, spread) / (self.max_spread_px * 1.25)
        else:
            spread_factor = 1.0
        spread_factor = float(np.clip(spread_factor, 0.0, 1.0))
        if jump is None or self.max_jump_px <= 0.0:
            jump_factor = 1.0
        else:
            jump_factor = 1.0 - max(0.0, jump) / (self.max_jump_px * 1.5)
            jump_factor = float(np.clip(jump_factor, 0.0, 1.0))
        return float(np.clip(0.55 * count_factor + 0.3 * spread_factor + 0.15 * jump_factor, 0.0, 1.0))

    def update(
        self,
        detections: Sequence[DotDetection],
        *,
        approx_depth_cm: float | None = None,
    ) -> tuple[dict[str, object] | None, dict[str, object]]:
        metrics: dict[str, object] = {
            'dots': len(detections),
            'used_dots': 0,
            'status': 'no_detections',
            'spread_px': None,
            'jump_px': None,
            'confidence': 0.0,
            'depth_cm': None,
            'depth_source': None,
            'points': [],
        }
        if not detections:
            self._register_miss()
            return None, metrics

        points = np.array([det.centroid for det in detections], dtype=np.float32)
        columns = np.array(
            [
                det.column if det.column is not None else "unknown"
                for det in detections
            ],
            dtype=object,
        )
        strengths = np.array(
            [max(det.strength, 0.0) for det in detections],
            dtype=np.float32,
        )
        has_left = bool(np.any(columns == "left"))
        has_right = bool(np.any(columns == "right"))
        depth_hint = approx_depth_cm if approx_depth_cm is not None else self.last_depth_cm
        allow_single = False
        allowed_column: str | None = None
        if not (has_left and has_right) and depth_hint is not None:
            present = "left" if has_left else ("right" if has_right else None)
            if present is not None:
                present_strengths = strengths[columns == present]
                if present_strengths.size and float(present_strengths.max()) >= CLUB_CORNER_COLUMN_MIN_STRENGTH:
                    allow_single = True
                    allowed_column = present
        if not allow_single and {"left", "right"} - {str(c) for c in columns}:
            metrics["status"] = "missing_columns"
            metrics["used_dots"] = int(points.shape[0])
            self._register_miss()
            return None, metrics
        weights = np.array(
            [max(det.area, 1.0) * max(det.brightness, 1.0) for det in detections],
            dtype=np.float32,
        ) * np.maximum(strengths, 1e-3)

        order = weights.argsort()[::-1]
        if order.size > self.max_candidates:
            order = order[: self.max_candidates]
        points = points[order]
        weights = weights[order]
        columns = columns[order]
        strengths = strengths[order]

        if points.shape[0] < self.min_dots:
            metrics['status'] = 'insufficient_dots'
            metrics['used_dots'] = int(points.shape[0])
            self._register_miss()
            return None, metrics

        mask = np.ones(points.shape[0], dtype=bool)
        center = self._weighted_center(points, weights, mask)
        if center is None:
            metrics['status'] = 'low_weight'
            self._register_miss()
            return None, metrics

        for _ in range(points.shape[0]):
            used = int(mask.sum())
            if used <= self.min_dots:
                break
            spread = self._cluster_spread(points, mask, center)
            if spread <= self.max_spread_px:
                break
            idxs = np.where(mask)[0]
            distances = np.linalg.norm(points[idxs] - center, axis=1)
            drop_idx = idxs[int(np.argmax(distances))]
            mask[drop_idx] = False
            center_candidate = self._weighted_center(points, weights, mask)
            if center_candidate is None:
                mask[drop_idx] = True
                break
            center = center_candidate

        used = int(mask.sum())
        center = self._weighted_center(points, weights, mask)
        used_columns = {
            str(col)
            for col, use_flag in zip(columns, mask)
            if use_flag and col not in (None, "unknown")
        }
        if not allow_single and {"left", "right"} - used_columns:
            metrics["status"] = "missing_columns"
            metrics["used_dots"] = used
            self._register_miss()
            return None, metrics
        if allow_single and allowed_column is not None and allowed_column not in used_columns:
            metrics["status"] = "missing_columns"
            metrics["used_dots"] = used
            self._register_miss()
            return None, metrics
        if center is None:
            metrics['status'] = 'low_weight'
            metrics['used_dots'] = used
            self._register_miss()
            return None, metrics

        raw_center = center.astype(np.float32, copy=False)
        if self.filtered_center is None:
            filtered = raw_center.copy()
        else:
            filtered = ((1.0 - self.ema_alpha) * self.filtered_center + self.ema_alpha * raw_center).astype(np.float32)

        spread_val = self._cluster_spread(points, mask, raw_center)
        if self.prev_center is None:
            jump_val = None
        else:
            jump_val = float(np.linalg.norm(raw_center - self.prev_center))

        confidence = self._confidence(used, spread_val, jump_val)

        metrics.update({
            'used_dots': used,
            'spread_px': float(spread_val),
            'jump_px': jump_val,
            'confidence': confidence,
            'raw_center': raw_center.copy(),
        })
        metrics['points'] = [
            {
                'x': float(pt[0]),
                'y': float(pt[1]),
                'column': str(col),
                'strength': float(strength),
                'used': bool(flag),
            }
            for pt, col, strength, flag in zip(points, columns, strengths, mask)
        ]

        if confidence < self.min_confidence:
            metrics['status'] = 'low_confidence'
            self._register_miss()
            return None, metrics

        self.filtered_center = filtered
        self.prev_center = raw_center.copy()
        self.miss_streak = 0

        geometry = self._geometry_from_points(points[mask], filtered, approx_depth_cm)
        if 'column_gap' in geometry and isinstance(geometry['column_gap'], dict):
            metrics['column_gap'] = geometry['column_gap']
        if 'orientation' in geometry and isinstance(geometry['orientation'], dict):
            metrics['orientation'] = geometry['orientation']
        if geometry.get('rejected'):
            metrics['status'] = str(geometry.get('reject_reason', 'invalid_geometry'))
            self._register_miss()
            return None, metrics
        metrics['depth_cm'] = geometry['depth_cm']
        metrics['depth_source'] = geometry['depth_source']
        if allow_single and allowed_column is not None:
            metrics['status'] = 'ok_single' if geometry['depth_cm'] is not None else 'ok_single_no_depth'
        else:
            metrics['status'] = 'ok' if geometry['depth_cm'] is not None else 'ok_no_depth'

        observation = {
            'center_px': geometry['center_px'],
            'raw_center': raw_center.copy(),
            'depth_cm': geometry['depth_cm'],
            'depth_source': geometry['depth_source'],
            'confidence': confidence,
            'used_dots': used,
            'dots': len(detections),
            'spread_px': float(spread_val),
            'jump_px': jump_val,
            'pairs': geometry['pairs'],
            'best_pair': geometry['best_pair'],
        }

        if geometry['depth_cm'] is not None:
            self.last_depth_cm = geometry['depth_cm']

        return observation, metrics

    def _validate_column_spacing(
        self,
        left_points: np.ndarray,
        right_points: np.ndarray,
        depth_hint: float | None,
    ) -> tuple[bool, dict[str, float]]:
        if left_points.size == 0 or right_points.size == 0 or depth_hint is None:
            return True, {}
        left_mean = float(left_points[:, 0].mean())
        right_mean = float(right_points[:, 0].mean())
        gap_px = abs(right_mean - left_mean)
        expected_px = (FOCAL_LENGTH * CLUBFACE_HORIZONTAL_SPACING_CM) / float(depth_hint)
        if not np.isfinite(expected_px) or expected_px <= 1e-3:
            return True, {}
        expected_px = max(expected_px, CLUB_CORNER_COLUMN_GAP_MIN_EXPECTED_PX)
        tolerance = max(CLUB_CORNER_COLUMN_GAP_ABS_TOL_PX, expected_px * CLUB_CORNER_COLUMN_GAP_REL_TOL)
        min_gap = max(2.0, expected_px - tolerance)
        max_gap = expected_px + tolerance
        info = {
            'gap_px': gap_px,
            'expected_px': expected_px,
            'min_px': min_gap,
            'max_px': max_gap,
        }
        if gap_px < min_gap:
            info['reason_code'] = -1.0
            return False, info
        if gap_px > max_gap:
            info['reason_code'] = 1.0
            return False, info
        info['reason_code'] = 0.0
        return True, info

    def _geometry_from_points(
        self,
        points: np.ndarray,
        filtered_center: np.ndarray,
        approx_depth_cm: float | None,
    ) -> dict[str, object]:
        if points.size == 0:
            return {
                'center_px': filtered_center.copy(),
                'depth_cm': None,
                'depth_source': None,
                'pairs': [],
                'best_pair': None,
            }

        left_idx, right_idx = self._split_columns(points)
        left_points = points[left_idx] if left_idx.size else np.empty((0, 2), dtype=np.float32)
        right_points = points[right_idx] if right_idx.size else np.empty((0, 2), dtype=np.float32)

        depth_hint = approx_depth_cm if approx_depth_cm is not None else self.last_depth_cm
        spacing_ok, spacing_info = self._validate_column_spacing(left_points, right_points, depth_hint)
        if not spacing_ok:
            return {
                'center_px': filtered_center.copy(),
                'depth_cm': None,
                'depth_source': None,
                'pairs': [],
                'best_pair': None,
                'rejected': True,
                'reject_reason': 'column_gap',
                'column_gap': spacing_info,
            }

        candidates = self._build_depth_candidates(left_points, right_points)
        best_pair: dict[str, object] | None = None
        depth_cm: float | None = None
        depth_source: str | None = None

        if candidates:
            best_pair = self._select_depth_candidate(candidates, approx_depth_cm)
            if best_pair is not None:
                depth_cm = float(best_pair['depth_cm'])
                depth_source = str(best_pair['type'])

        if depth_cm is None:
            if approx_depth_cm is not None:
                depth_cm = float(approx_depth_cm)
                depth_source = 'approx'
            elif self.last_depth_cm is not None:
                depth_cm = float(self.last_depth_cm)
                depth_source = 'history'

        if depth_cm is not None:
            spacing_ok_depth, spacing_info = self._validate_column_spacing(left_points, right_points, depth_cm)
            if not spacing_ok_depth:
                return {
                    'center_px': filtered_center.copy(),
                    'depth_cm': None,
                    'depth_source': None,
                    'pairs': candidates,
                    'best_pair': best_pair,
                    'rejected': True,
                    'reject_reason': 'column_gap_depth',
                    'column_gap': spacing_info,
                }

        center_px = filtered_center.copy()

        top_left = float(left_points[:, 1].min()) if left_points.size else None
        top_right = float(right_points[:, 1].min()) if right_points.size else None
        bottom_left = float(left_points[:, 1].max()) if left_points.size else None
        bottom_right = float(right_points[:, 1].max()) if right_points.size else None
        span_left = max((bottom_left - top_left), 1.0) if bottom_left is not None else 0.0
        span_right = max((bottom_right - top_right), 1.0) if bottom_right is not None else 0.0
        orientation_info: dict[str, float] = {}
        if top_left is not None:
            orientation_info['top_left_px'] = top_left
        if top_right is not None:
            orientation_info['top_right_px'] = top_right
        if top_left is not None and top_right is not None:
            orientation_info['delta_top_px'] = top_right - top_left
            orientation_info['tolerance_px'] = CLUB_CORNER_COLUMN_TOP_TOL
            if top_left > top_right + CLUB_CORNER_COLUMN_TOP_TOL:
                return {
                    'center_px': filtered_center.copy(),
                    'depth_cm': None,
                    'depth_source': None,
                    'pairs': candidates,
                    'best_pair': best_pair,
                    'rejected': True,
                    'reject_reason': 'column_orientation',
                    'column_gap': spacing_info,
                    'orientation': orientation_info,
                }
        if top_left is not None and top_right is not None:
            if top_left + CLUB_CENTER_TOP_ALIGNMENT_TOL < top_right:
                center_px[1] = float(max(0.0, top_right - CLUB_CENTER_SINGLE_COLUMN_UPSHIFT * span_right))
            elif top_right + CLUB_CENTER_TOP_ALIGNMENT_TOL < top_left:
                center_px[1] = float(max(0.0, top_left - CLUB_CENTER_SINGLE_COLUMN_UPSHIFT * span_left))
            else:
                blended = 0.5 * (top_left + top_right)
                avg_span = 0.5 * (span_left + span_right)
                center_px[1] = float(max(0.0, blended - CLUB_CENTER_SINGLE_COLUMN_UPSHIFT * avg_span))
        elif top_left is not None:
            center_px[1] = float(max(0.0, top_left - CLUB_CENTER_SINGLE_COLUMN_UPSHIFT * span_left))
        elif top_right is not None:
            center_px[1] = float(max(0.0, top_right - CLUB_CENTER_SINGLE_COLUMN_UPSHIFT * span_right))

        if depth_cm is not None:
            if left_points.size:
                left_x = float(left_points[:, 0].mean())
                if right_points.size:
                    right_x = float(right_points[:, 0].mean())
                    center_px[0] = left_x + (right_x - left_x) * 0.5
                else:
                    offset_px = (FOCAL_LENGTH * CLUBFACE_CENTER_OFFSET_CM) / depth_cm
                    center_px[0] = left_x + offset_px
            elif right_points.size:
                right_x = float(right_points[:, 0].mean())
                offset_px = (FOCAL_LENGTH * CLUBFACE_CENTER_OFFSET_CM) / depth_cm
                center_px[0] = right_x - offset_px
        else:
            if left_points.size:
                center_px[0] = float(left_points[:, 0].mean())
            elif right_points.size:
                center_px[0] = float(right_points[:, 0].mean())

        return {
            'center_px': center_px.astype(np.float32),
            'depth_cm': depth_cm,
            'depth_source': depth_source,
            'pairs': candidates,
            'best_pair': best_pair,
            'column_gap': spacing_info,
            'rejected': False,
            'orientation': orientation_info,
        }

    @staticmethod
    def _split_columns(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if points.shape[0] <= 1:
            idx = np.arange(points.shape[0])
            return idx, np.empty(0, dtype=int)
        sorted_idx = np.argsort(points[:, 0])
        x_sorted = points[sorted_idx, 0]
        gaps = np.diff(x_sorted)
        if gaps.size == 0:
            return sorted_idx, np.empty(0, dtype=int)
        max_gap_idx = int(np.argmax(gaps))
        if gaps[max_gap_idx] >= CLUBFACE_COLUMN_SPLIT_PX:
            split = max_gap_idx + 1
            left = sorted_idx[:split]
            right = sorted_idx[split:]
        else:
            left = sorted_idx
            right = np.empty(0, dtype=int)
        return left, right

    def _build_depth_candidates(
        self,
        left_points: np.ndarray,
        right_points: np.ndarray,
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        if left_points.shape[0] >= 2:
            order = np.argsort(left_points[:, 1])
            ordered = left_points[order]
            for i in range(ordered.shape[0]):
                for j in range(i + 1, ordered.shape[0]):
                    steps = j - i
                    real_cm = steps * CLUBFACE_VERTICAL_SPACING_CM
                    if real_cm <= 0.0:
                        continue
                    dist_px = float(np.linalg.norm(ordered[i] - ordered[j]))
                    if dist_px < 1.0:
                        continue
                    depth_cm = (FOCAL_LENGTH * real_cm) / dist_px
                    candidates.append(
                        {
                            'depth_cm': depth_cm,
                            'pixels': dist_px,
                            'type': f'vertical_left_{steps}',
                            'points': (ordered[i], ordered[j]),
                            'steps': steps,
                        }
                    )
        if left_points.shape[0] >= 1 and right_points.shape[0] >= 1:
            pair_candidates: list[dict[str, object]] = []
            for lp in left_points:
                for rp in right_points:
                    vertical_delta = abs(float(lp[1]) - float(rp[1]))
                    if vertical_delta > CLUBFACE_VERTICAL_ALIGNMENT_PX:
                        continue
                    dist_px = float(np.linalg.norm(lp - rp))
                    if dist_px < 1.0:
                        continue
                    depth_cm = (FOCAL_LENGTH * CLUBFACE_HORIZONTAL_SPACING_CM) / dist_px
                    pair_candidates.append(
                        {
                            'depth_cm': depth_cm,
                            'pixels': dist_px,
                            'type': 'horizontal_pair',
                            'points': (lp, rp),
                            'vertical_delta': vertical_delta,
                        }
                    )
            if pair_candidates:
                candidates.extend(pair_candidates)
            else:
                right_point = right_points.mean(axis=0)
                for lp in left_points:
                    dist_px = float(np.linalg.norm(lp - right_point))
                    if dist_px < 1.0:
                        continue
                    depth_cm = (FOCAL_LENGTH * CLUBFACE_HORIZONTAL_SPACING_CM) / dist_px
                    candidates.append(
                        {
                            'depth_cm': depth_cm,
                            'pixels': dist_px,
                            'type': 'horizontal_mean',
                            'points': (lp, right_point.copy()),
                            'vertical_delta': abs(float(lp[1]) - float(right_point[1])),
                        }
                    )
        if right_points.shape[0] >= 2:
            order_r = np.argsort(right_points[:, 1])
            ordered_r = right_points[order_r]
            for i in range(ordered_r.shape[0]):
                for j in range(i + 1, ordered_r.shape[0]):
                    steps = j - i
                    real_cm = steps * CLUBFACE_VERTICAL_SPACING_CM
                    if real_cm <= 0.0:
                        continue
                    dist_px = float(np.linalg.norm(ordered_r[i] - ordered_r[j]))
                    if dist_px < 1.0:
                        continue
                    depth_cm = (FOCAL_LENGTH * real_cm) / dist_px
                    candidates.append(
                        {
                            'depth_cm': depth_cm,
                            'pixels': dist_px,
                            'type': f'vertical_right_{steps}',
                            'points': (ordered_r[i], ordered_r[j]),
                            'steps': steps,
                        }
                    )
        return candidates

    def _select_depth_candidate(
        self,
        candidates: list[dict[str, object]],
        approx_depth_cm: float | None,
    ) -> dict[str, object] | None:
        if not candidates:
            return None

        def priority(candidate: dict[str, object]) -> int:
            ctype = str(candidate['type'])
            if ctype.startswith('horizontal'):
                return 0
            if ctype.startswith('vertical') and ctype.endswith('_1'):
                return 1
            return 2

        ref = approx_depth_cm if approx_depth_cm is not None else self.last_depth_cm
        filtered = [c for c in candidates if CLUBFACE_DEPTH_MIN_CM <= c['depth_cm'] <= CLUBFACE_DEPTH_MAX_CM]
        pool = filtered if filtered else candidates
        if ref is not None:
            pool.sort(
                key=lambda c: (
                    priority(c),
                    abs(float(c['depth_cm']) - ref),
                    float(c.get('vertical_delta', 0.0)),
                )
            )
        else:
            pool.sort(
                key=lambda c: (
                    priority(c),
                    -float(c['pixels']),
                    float(c.get('vertical_delta', 0.0)),
                )
            )
        return pool[0] if pool else None

def detect_reflective_dots(
    off_frame: np.ndarray | None, on_frame: np.ndarray | None
) -> list[DotDetection]:
    if on_frame is None or on_frame.size == 0:
        return []

    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2 or image.shape[-1] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    on_bgr = _ensure_bgr(on_frame)
    height, width = on_bgr.shape[:2]
    if height == 0 or width == 0:
        return []

    resized = cv2.resize(
        on_bgr,
        (CLUB_CORNER_RESIZE_WIDTH, CLUB_CORNER_RESIZE_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    scale_x = width / float(CLUB_CORNER_RESIZE_WIDTH)
    scale_y = height / float(CLUB_CORNER_RESIZE_HEIGHT)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    if CLAHE is not None:
        gray = CLAHE.apply(gray)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=CLUB_CORNER_GFTT_MAX_CORNERS,
        qualityLevel=CLUB_CORNER_GFTT_QUALITY,
        minDistance=CLUB_CORNER_GFTT_MIN_DISTANCE,
        blockSize=CLUB_HARRIS_BLOCK_SIZE,
        useHarrisDetector=True,
        k=CLUB_HARRIS_K,
    )
    if corners is None or corners.size == 0:
        return []

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01,
    )
    cv2.cornerSubPix(
        gray,
        corners,
        CLUB_CORNER_SUBPIX_WINDOW,
        CLUB_CORNER_SUBPIX_ZERO_ZONE,
        criteria,
    )
    corners = corners.reshape(-1, 2)

    gray_float = np.float32(gray)
    harris_response = cv2.cornerHarris(gray_float, CLUB_HARRIS_BLOCK_SIZE, CLUB_HARRIS_KSIZE, CLUB_HARRIS_K)
    harris_response = cv2.GaussianBlur(harris_response, (3, 3), 0)
    np.maximum(harris_response, 0, out=harris_response)
    response_max = float(harris_response.max())

    height_thresh = min(float(height) * DOT_MIN_Y_FRACTION, DOT_MIN_Y_PX)
    if height < 200:
        height_thresh *= float(height) / 200.0
    height_thresh = max(0.0, height_thresh)

    detections: list[DotDetection] = []
    kept_points: list[np.ndarray] = []
    avg_scale = 0.5 * (scale_x + scale_y)
    radius = max(3.0 * avg_scale, 1.0)
    area = math.pi * (radius ** 2)
    for x_res, y_res in corners:
        if not (np.isfinite(x_res) and np.isfinite(y_res)):
            continue
        cx = float(x_res * scale_x)
        cy = float(y_res * scale_y)
        if cy < height_thresh:
            continue
        if kept_points:
            distances = np.linalg.norm(
                np.array(kept_points, dtype=np.float32)
                - np.array([cx, cy], dtype=np.float32),
                axis=1,
            )
            if float(distances.min()) < CLUB_CORNER_DUPLICATE_DISTANCE_PX:
                continue
        xi = int(np.clip(round(x_res), 0, harris_response.shape[1] - 1))
        yi = int(np.clip(round(y_res), 0, harris_response.shape[0] - 1))
        response = float(harris_response[yi, xi])
        strength = response / response_max if response_max > 1e-6 else 0.0
        if strength < CLUB_CORNER_MIN_RESPONSE:
            continue
        contrast = _corner_patch_contrast(gray, x_res, y_res, CLUB_CORNER_CONTRAST_WINDOW)
        if contrast < CLUB_CORNER_MIN_CONTRAST:
            continue
        brightness = float(np.clip(strength, 0.0, 1.0) * 255.0)
        detections.append(
            DotDetection(
                centroid=np.array([cx, cy], dtype=np.float32),
                area=area,
                brightness=brightness,
                circularity=1.0,
                strength=strength,
            )
        )
        kept_points.append(np.array([cx, cy], dtype=np.float32))
        if len(detections) >= CLUB_CORNER_MAX_TOTAL:
            break

    if len(detections) < DOT_MIN_LEFT_COLUMN + DOT_MIN_RIGHT_COLUMN:
        return []

    return _finalize_corner_columns(detections, width, height)


def _finalize_corner_columns(
    detections: list[DotDetection],
    width: float,
    height: float,
) -> list[DotDetection]:
    if len(detections) < DOT_MIN_LEFT_COLUMN + DOT_MIN_RIGHT_COLUMN:
        return []
    coords = np.array([d.centroid for d in detections], dtype=np.float32)
    if coords.shape[0] <= 1:
        return []
    sorted_idx = np.argsort(coords[:, 0])
    gaps = np.diff(coords[sorted_idx, 0])
    if gaps.size == 0:
        return []
    max_gap = float(gaps.max())
    if max_gap < CLUBFACE_COLUMN_SPLIT_PX:
        return []
    split = int(np.argmax(gaps) + 1)
    left_idx = sorted_idx[:split]
    right_idx = sorted_idx[split:]
    if left_idx.size < DOT_MIN_LEFT_COLUMN or right_idx.size < DOT_MIN_RIGHT_COLUMN:
        return []

    def _filter(indices: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            return indices
        pts = coords[indices]
        center_x = float(np.median(pts[:, 0]))
        deviation = np.abs(pts[:, 0] - center_x)
        keep = deviation <= CLUB_CORNER_COLUMN_MAX_DEVIATION_PX
        if not np.any(keep):
            keep = deviation <= CLUB_CORNER_COLUMN_MAX_DEVIATION_PX * 1.5
        return indices[keep]

    left_idx = _filter(left_idx)
    right_idx = _filter(right_idx)
    if left_idx.size < DOT_MIN_LEFT_COLUMN or right_idx.size < DOT_MIN_RIGHT_COLUMN:
        return []

    column_width_limit = max(CLUB_CORNER_COLUMN_MAX_WIDTH_PX, float(width) * 0.03)
    left_pts = coords[left_idx]
    right_pts = coords[right_idx]
    if left_pts.shape[0] >= 2:
        if float(left_pts[:, 0].ptp()) > column_width_limit:
            return []
        if float(left_pts[:, 1].ptp()) < max(
            CLUB_CORNER_MIN_VERTICAL_SPREAD_PX,
            height * CLUB_CORNER_HEIGHT_SCALE,
        ):
            return []
    if right_pts.shape[0] >= 2:
        if float(right_pts[:, 0].ptp()) > column_width_limit:
            return []
        if float(right_pts[:, 1].ptp()) < max(
            CLUB_CORNER_MIN_VERTICAL_SPREAD_PX,
            height * CLUB_CORNER_HEIGHT_SCALE * 0.8,
        ):
            return []

    left_strength = max(float(detections[int(i)].strength) for i in left_idx)
    right_strength = max(float(detections[int(i)].strength) for i in right_idx)
    if (
        left_strength < CLUB_CORNER_COLUMN_MIN_STRENGTH
        or right_strength < CLUB_CORNER_COLUMN_MIN_STRENGTH
    ):
        return []

    left_idx_sorted = sorted(left_idx, key=lambda i: float(detections[int(i)].strength), reverse=True)
    right_idx_sorted = sorted(right_idx, key=lambda i: float(detections[int(i)].strength), reverse=True)
    left_idx_sorted = np.array(left_idx_sorted[:CLUB_CORNER_MAX_PER_LEFT], dtype=int)
    right_idx_sorted = np.array(right_idx_sorted[:CLUB_CORNER_MAX_PER_RIGHT], dtype=int)
    if left_idx_sorted.size == 0 or right_idx_sorted.size == 0:
        return []

    result: list[DotDetection] = []
    for idx in left_idx_sorted:
        det = detections[int(idx)]
        result.append(
            DotDetection(
                centroid=det.centroid.copy(),
                area=det.area,
                brightness=det.brightness,
                circularity=det.circularity,
                column="left",
                strength=det.strength,
            )
        )
    for idx in right_idx_sorted:
        det = detections[int(idx)]
        result.append(
            DotDetection(
                centroid=det.centroid.copy(),
                area=det.area,
                brightness=det.brightness,
                circularity=det.circularity,
                column="right",
                strength=det.strength,
            )
        )

    result.sort(key=lambda d: d.brightness, reverse=True)
    if len(result) > DOT_MAX_DETECTIONS:
        result = result[:DOT_MAX_DETECTIONS]
    return result
def _polyfit_with_mask(
    times: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    degree: int,
) -> np.ndarray | None:
    """Return polynomial coefficients for the selected ``times``/``values`` subset."""
    count = int(np.count_nonzero(mask))
    if count == 0:
        return None
    deg = int(max(0, min(degree, count - 1)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            coeffs = np.polyfit(times[mask], values[mask], deg)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return coeffs


def _robust_polyfit(
    times: np.ndarray,
    values: np.ndarray,
    *,
    degree: int,
    max_iter: int = 5,
    sigma: float = 2.5,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Robust polynomial fit that iteratively masks outliers using a MAD threshold."""
    finite_mask = np.isfinite(values)
    if int(finite_mask.sum()) == 0:
        return None, np.full_like(values, np.nan, dtype=np.float64), finite_mask

    current_mask = finite_mask.copy()
    coeffs = _polyfit_with_mask(times, values, current_mask, degree)
    if coeffs is None:
        return None, np.full_like(values, np.nan, dtype=np.float64), current_mask

    for _ in range(max_iter):
        fitted = np.polyval(coeffs, times)
        residuals = values - fitted
        inlier_residuals = residuals[current_mask]
        if inlier_residuals.size == 0:
            break
        scale = float(np.median(np.abs(inlier_residuals)))
        if not np.isfinite(scale) or scale < 1e-6:
            break
        threshold = sigma * 1.4826 * scale + 1e-6
        new_mask = current_mask & (np.abs(residuals) <= threshold)
        if int(new_mask.sum()) == int(current_mask.sum()):
            break
        current_mask = new_mask
        coeffs = _polyfit_with_mask(times, values, current_mask, degree)
        if coeffs is None:
            break

    if coeffs is None:
        coeffs = _polyfit_with_mask(times, values, finite_mask, 0)
    fitted = np.polyval(coeffs, times) if coeffs is not None else np.copy(values)
    return coeffs, fitted, current_mask


def _refine_mask_with_residuals(
    times: np.ndarray,
    axis_values: dict[str, np.ndarray],
    base_mask: np.ndarray,
    degree: int,
) -> np.ndarray:
    mask = base_mask.copy()
    changed = False
    for _ in range(CLUB_TRAJECTORY_MASK_ITERS):
        if int(mask.sum()) <= max(degree, 1):
            break
        coeffs: dict[str, np.ndarray] = {}
        for axis, values in axis_values.items():
            coeff = _polyfit_with_mask(times, values, mask, degree)
            if coeff is None:
                return mask if changed else base_mask
            coeffs[axis] = coeff
        predictions = {axis: np.polyval(coeff, times) for axis, coeff in coeffs.items()}
        residuals = {axis: axis_values[axis] - predictions[axis] for axis in axis_values}
        new_mask = mask.copy()
        for axis, res in residuals.items():
            masked_res = res[mask]
            if masked_res.size == 0:
                continue
            abs_res = np.abs(masked_res)
            mad = float(np.median(np.abs(abs_res - np.median(abs_res))))
            threshold = CLUB_TRAJECTORY_RESIDUAL_SIGMA * 1.4826 * mad + 1e-6
            threshold = max(CLUB_TRAJECTORY_AXIS_MIN_RESIDUAL.get(axis, 0.5), threshold)
            new_mask &= np.abs(res) <= threshold
        residual_matrix = np.column_stack([residuals.get('x'), residuals.get('y'), residuals.get('z')])
        if residual_matrix.size:
            residual_norm = np.linalg.norm(residual_matrix, axis=1)
            masked_norm = residual_norm[mask]
            if masked_norm.size:
                mad_norm = float(np.median(np.abs(masked_norm - np.median(masked_norm))))
                norm_thresh = CLUB_TRAJECTORY_NORM_SIGMA * 1.4826 * mad_norm + 1e-6
                norm_thresh = max(CLUB_TRAJECTORY_MIN_NORM, norm_thresh)
                new_mask &= residual_norm <= norm_thresh
        if int(new_mask.sum()) <= max(degree, 1):
            break
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
        changed = True
    return mask if changed else base_mask

def _world_to_pixel(x_cm: float, y_cm: float, z_cm: float) -> tuple[float, float] | None:
    depth_cm = float(z_cm + CLUBFACE_Z_OFFSET_CM)
    if depth_cm <= 1e-3 or not np.isfinite(depth_cm):
        return None
    u = x_cm * FX / depth_cm + CX
    v = y_cm * FY / depth_cm + CY
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return float(u), float(v)


def _coords_to_pixel_points(
    coords: list[dict[str, float]],
    original_times: set[float],
) -> list[dict[str, float]]:
    """Project trajectory coordinates into pixel space."""
    points: list[dict[str, float]] = []
    for entry in coords:
        try:
            time_val = round(float(entry["time"]), 3)
            x_cm = float(entry["x"])
            y_cm = float(entry["y"])
            z_cm = float(entry["z"])
        except (KeyError, TypeError, ValueError):
            continue
        pixel = _world_to_pixel(x_cm, y_cm, z_cm)
        if pixel is None:
            continue
        u, v = pixel
        points.append(
            {
                "time": time_val,
                "u": u,
                "v": v,
                "is_original": time_val in original_times,
            }
        )
    return points


def refine_clubface_trajectory(coords: list[dict[str, float]]) -> RefinedTrajectory:
    """Remove outliers from clubface samples and interpolate gaps along a smooth curve."""
    original_times = {
        round(float(entry["time"]), 3)
        for entry in coords
        if "time" in entry
    }
    if len(coords) < 4:
        pixel_points = _coords_to_pixel_points(coords, original_times)
        return RefinedTrajectory(coords, original_times.copy(), original_times, pixel_points)

    time_groups: dict[float, list[tuple[float, float, float]]] = {}
    for entry in coords:
        try:
            t = float(entry["time"])
            x_val = float(entry["x"])
            y_val = float(entry["y"])
            z_val = float(entry["z"])
        except (KeyError, TypeError, ValueError):
            continue
        time_groups.setdefault(t, []).append((x_val, y_val, z_val))

    if len(time_groups) < 3:
        pixel_points = _coords_to_pixel_points(coords, original_times)
        return RefinedTrajectory(coords, original_times.copy(), original_times, pixel_points)

    sorted_times = [float(t) for t in sorted(time_groups.keys())]
    times = np.array(sorted_times, dtype=np.float64)
    xs = np.array(
        [float(np.mean([p[0] for p in time_groups[t]])) for t in sorted_times],
        dtype=np.float64,
    )
    ys = np.array(
        [float(np.mean([p[1] for p in time_groups[t]])) for t in sorted_times],
        dtype=np.float64,
    )
    zs = np.array(
        [float(np.mean([p[2] for p in time_groups[t]])) for t in sorted_times],
        dtype=np.float64,
    )

    degree_target = 2 if times.size >= 3 else max(1, times.size - 1)
    coeff_x, _, mask_x = _robust_polyfit(times, xs, degree=degree_target)
    coeff_y, _, mask_y = _robust_polyfit(times, ys, degree=degree_target)
    coeff_z, _, mask_z = _robust_polyfit(times, zs, degree=degree_target)
    if coeff_x is None or coeff_y is None or coeff_z is None:
        pixel_points = _coords_to_pixel_points(coords, original_times)
        return RefinedTrajectory(coords, original_times.copy(), original_times, pixel_points)

    combined_mask = mask_x & mask_y & mask_z
    if int(combined_mask.sum()) < 3:
        combined_mask = mask_x | mask_y | mask_z
    if int(combined_mask.sum()) < 2:
        pixel_points = _coords_to_pixel_points(coords, original_times)
        return RefinedTrajectory(coords, original_times.copy(), original_times, pixel_points)

    axis_values = {"x": xs, "y": ys, "z": zs}
    refined_mask = _refine_mask_with_residuals(times, axis_values, combined_mask, degree_target)
    if int(refined_mask.sum()) >= max(degree_target + 1, 3):
        combined_mask = refined_mask
    final_coeffs: dict[str, np.ndarray] = {}
    for axis, values in axis_values.items():
        coeffs = _polyfit_with_mask(times, values, combined_mask, degree_target)
        if coeffs is None:
            pixel_points = _coords_to_pixel_points(coords, original_times)
            return RefinedTrajectory(coords, original_times.copy(), original_times, pixel_points)
        final_coeffs[axis] = coeffs

    combined_times: list[float] = sorted_times.copy()
    if times.size > 1:
        diffs = np.diff(times)
        positive_diffs = diffs[diffs > 1e-6]
        dt = float(np.median(positive_diffs)) if positive_diffs.size else None
    else:
        dt = None
    if dt is not None and dt > 1e-6:
        for prev, curr in zip(times[:-1], times[1:]):
            gap = curr - prev
            if gap <= 1.5 * dt:
                continue
            num_missing = max(0, int(round(gap / dt)) - 1)
            for step in range(1, num_missing + 1):
                combined_times.append(float(prev + step * dt))

    combined_times_arr = np.unique(
        np.round(np.array(combined_times, dtype=np.float64), 6)
    )
    x_curve = np.polyval(final_coeffs["x"], combined_times_arr)
    y_curve = np.polyval(final_coeffs["y"], combined_times_arr)
    z_curve = np.polyval(final_coeffs["z"], combined_times_arr)

    refined_coords: list[dict[str, float]] = []
    last_time: float | None = None
    for t, x_val, y_val, z_val in zip(combined_times_arr, x_curve, y_curve, z_curve):
        if not (
            np.isfinite(t)
            and np.isfinite(x_val)
            and np.isfinite(y_val)
            and np.isfinite(z_val)
        ):
            continue
        time_out = round(float(t), 3)
        if last_time is not None and abs(time_out - last_time) <= 1e-3:
            if refined_coords:
                refined_coords[-1]["x"] = round(float(x_val), 2)
                refined_coords[-1]["y"] = round(float(y_val), 2)
                refined_coords[-1]["z"] = round(float(z_val), 2)
            continue
        refined_coords.append(
            {
                "time": time_out,
                "x": round(float(x_val), 2),
                "y": round(float(y_val), 2),
                "z": round(float(z_val), 2),
            }
        )
        last_time = time_out

    inlier_times = {
        round(float(sorted_times[idx]), 3)
        for idx, flag in enumerate(combined_mask)
        if flag
    }
    if not refined_coords:
        refined_coords = coords
    pixel_points = _coords_to_pixel_points(refined_coords, original_times)
    return RefinedTrajectory(refined_coords, inlier_times, original_times, pixel_points)


def _annotate_clubface_frames(
    frames_dir: str,
    samples_by_frame: dict[int, list[dict[str, object]]],
    trajectory: RefinedTrajectory,
) -> None:
    """Overlay refined clubface trajectory and point status on saved frame images."""
    if not frames_dir or not trajectory.coords:
        return
    if not os.path.isdir(frames_dir):
        return

    interpolated_points = []
    outlier_times = trajectory.original_times - trajectory.inlier_times
    frame_entries: list[tuple[int, str]] = []
    for name in os.listdir(frames_dir):
        if not (name.startswith("frame_") and name.endswith(".png")):
            continue
        try:
            frame_idx = int(name[6:-4])
        except ValueError:
            continue
        frame_entries.append((frame_idx, os.path.join(frames_dir, name)))
    if not frame_entries:
        return

    for frame_idx, frame_path in sorted(frame_entries):
        image = cv2.imread(frame_path)
        if image is None:
            continue
        height, width = image.shape[:2]

        samples = samples_by_frame.get(frame_idx, [])
        for sample in samples:
            center = sample.get("center_px")
            time_val = sample.get("time")
            detections = sample.get("points") or []
            if center is None or time_val is None:
                continue
            try:
                u = float(center[0])
                v = float(center[1])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            if not (0 <= u < width and 0 <= v < height):
                continue
            is_inlier = time_val in trajectory.inlier_times
            color = (0, 255, 0) if is_inlier else (0, 0, 255)
            cv2.circle(
                image,
                (int(round(u)), int(round(v))),
                5,
                color,
                -1,
                cv2.LINE_AA,
            )
            for det in detections:
                try:
                    dx = float(det.get("x", float("nan")))
                    dy = float(det.get("y", float("nan")))
                except (TypeError, ValueError):
                    continue
                if not (np.isfinite(dx) and np.isfinite(dy)):
                    continue
                if not (0 <= dx < width and 0 <= dy < height):
                    continue
                column_label = str(det.get("column", "unknown")).lower()
                used_flag = bool(det.get("used", False))
                if column_label == "left":
                    det_color = (0, 165, 255)
                    label_text = "L"
                elif column_label == "right":
                    det_color = (0, 255, 255)
                    label_text = "R"
                else:
                    det_color = (200, 200, 200)
                    label_text = "?"
                radius = 4 if used_flag else 3
                center_point = (int(round(dx)), int(round(dy)))
                cv2.circle(
                    image,
                    center_point,
                    radius,
                    det_color,
                    -1,
                    cv2.LINE_AA,
                )
                label_org = (center_point[0] + 6, center_point[1] - 6)
                cv2.putText(
                    image,
                    label_text,
                    label_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    label_text,
                    label_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    det_color,
                    1,
                    cv2.LINE_AA,
                )
            if not is_inlier and time_val in outlier_times:
                cv2.drawMarker(
                    image,
                    (int(round(u)), int(round(v))),
                    (0, 0, 255),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=10,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )

        cv2.imwrite(frame_path, image)


def bbox_to_ball_metrics(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """Return center, radius and distance estimates for a bounding box."""

    w = float(x2 - x1)
    h = float(y2 - y1)
    raw_radius = (w + h) / 4.0
    radius_for_distance = max(raw_radius, 1e-6)
    cx = float(x1 + x2) / 2.0
    cy = float(y1 + y2) / 2.0
    distance = FOCAL_LENGTH * ACTUAL_BALL_RADIUS / radius_for_distance
    return cx, cy, raw_radius, distance


def bbox_within_image(
    bbox: tuple[float, float, float, float],
    width: float,
    height: float,
    *,
    margin: float = EDGE_MARGIN_PX,
) -> bool:
    """Return ``True`` when ``bbox`` stays inside the image bounds with ``margin`` padding."""

    x1, y1, x2, y2 = bbox
    return (
        x1 > margin
        and y1 > margin
        and x2 < width - margin
        and y2 < height - margin
    )


def preprocess_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enhance ``frame`` for low light and return both color and gray images."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = CLAHE.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray = CLAHE.apply(gray)
    if USE_BLUR:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return enhanced, gray


class TFLiteBallDetector:
    """Thin wrapper around the exported YOLOv8 TFLite model with custom post-processing."""

    def __init__(
        self,
        model_path: str,
        *,
        conf_threshold: float = 0.01,
        iou_threshold: float = 0.4,
        max_detections: int = 10,
    ) -> None:
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        _, self.input_height, self.input_width, _ = self.input_details["shape"]
        self.input_dtype = self.input_details["dtype"]
        q_scale, q_zero_point = self.input_details.get("quantization", (0.0, 0))
        self.input_scale = float(q_scale) if isinstance(q_scale, (np.ndarray, list)) else float(q_scale)
        self.input_zero_point = (
            int(q_zero_point[0])
            if isinstance(q_zero_point, (np.ndarray, list))
            else int(q_zero_point)
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run inference on ``frame`` and return a list of detections sorted by confidence."""

        letterboxed, ratio, pad = self._letterbox(frame, (self.input_height, self.input_width))
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        input_tensor = rgb.astype(np.float32) / 255.0
        if self.input_dtype != np.float32:
            if self.input_scale == 0.0:
                raise RuntimeError("Quantized model lacks scale factor")
            input_tensor = np.round(input_tensor / self.input_scale + self.input_zero_point).astype(
                self.input_dtype
            )
        else:
            input_tensor = input_tensor.astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        self.interpreter.set_tensor(self.input_details["index"], input_tensor)
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.output_details["index"])
        preds = np.squeeze(raw_output, axis=0).transpose(1, 0)  # (anchors, 5)
        if preds.size == 0:
            return []
        boxes_xywh = preds[:, :4]
        scores = preds[:, 4]
        mask = scores >= self.conf_threshold
        if not np.any(mask):
            return []
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        left, top = pad
        boxes_xyxy[:, [0, 2]] -= left
        boxes_xyxy[:, [1, 3]] -= top
        if ratio == 0:
            raise RuntimeError("Letterbox ratio is zero")
        inv_ratio = 1.0 / ratio
        boxes_xyxy[:, [0, 2]] *= inv_ratio
        boxes_xyxy[:, [1, 3]] *= inv_ratio
        h, w = frame.shape[:2]
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0.0, w - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0.0, h - 1)
        keep = self._nms(boxes_xyxy, scores)
        detections = []
        for idx in keep[: self.max_detections]:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            detections.append(
                {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "score": float(scores[idx]),
                }
            )
        return detections

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        x, y, w, h = boxes.T
        half_w = w / 2.0
        half_h = h / 2.0
        x1 = x - half_w
        y1 = y - half_h
        x2 = x + half_w
        y2 = y + half_h
        return np.stack((x1, y1, x2, y2), axis=1)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            ious = self._iou(boxes[i], boxes[rest])
            order = rest[ious <= self.iou_threshold]
        return keep

    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.empty(0)
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter = inter_w * inter_h
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter + 1e-6
        return inter / union

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        new_shape: tuple[int, int],
        color: tuple[int, int, int] = (0, 0, 0),
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Resize ``image`` with unchanged aspect ratio using padding."""

        shape = image.shape[:2]  # h, w
        target_h, target_w = new_shape
        if shape[0] == 0 or shape[1] == 0:
            raise ValueError("Invalid frame shape for letterbox")
        ratio = min(target_h / shape[0], target_w / shape[1])
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        if new_unpad[0] != shape[1] or new_unpad[1] != shape[0]:
            resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = image
        dw = target_w - new_unpad[0]
        dh = target_h - new_unpad[1]
        dw /= 2
        dh /= 2
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        if padded.shape[0] != target_h or padded.shape[1] != target_w:
            padded = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return padded, ratio, (left, top)



def find_motion_window(
    video_path: str,
    detector: TFLiteBallDetector,
    *,
    pad_frames: int = MAX_MOTION_FRAMES,
    max_frames: int = MAX_MOTION_FRAMES,
    confirm_radius: int = 3,
    score_threshold: float = MOTION_WINDOW_SCORE_THRESHOLD,
    debug: bool = False,
) -> tuple[int, int, bool, dict[str, int | bool | None]]:
    """Return a motion window around the last confident ball sighting.

    The search prioritises late frames by sampling the video with progressively
    finer strides until the first positive detection is found, then refines the
    exact exit frame with a short forward scan. The helper records lightweight
    stats describing how many frames were actually decoded so we can validate
    efficiency claims."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    miss_limit = (
        MOTION_WINDOW_EARLY_EXIT_MISSES
        if MOTION_WINDOW_EARLY_EXIT_MISSES is not None
        else max(confirm_radius * 3, 9)
    )
    max_eval_budget = max(
        total * MOTION_WINDOW_MAX_SCAN_MULTIPLIER,
        total + 512,
        MOTION_WINDOW_MIN_SCAN_BUDGET,
    )
    decode_failure_limit = max(
        MOTION_WINDOW_DECODE_FAILURE_LIMIT,
        confirm_radius * 4,
    )

    stats: dict[str, int | bool | None] = {
        "total_frames": total,
        "frames_evaluated": 0,
        "frames_decoded": 0,
        "frames_with_ball": 0,
        "detector_runs": 0,
        "seeks": 0,
        "coarse_step": None,
        "coarse_frames_scanned": 0,
        "coarse_found_frame": None,
        "coarse_false_frame": None,
        "fallback_full_scan": False,
        "refine_frames": 0,
        "refine_last_frame": None,
        "miss_limit": miss_limit,
        "max_eval_budget": int(max_eval_budget),
        "decode_failure_limit": int(decode_failure_limit),
        "ball_always_visible": False,
        "ball_never_observed": False,
    }

    detection_cache: dict[int, dict | None] = {}
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    current_pos: int | None = None
    decode_failures = 0

    def finalize_stats() -> dict[str, int | bool | None]:
        snapshot = dict(stats)
        snapshot["frames_processed"] = len(detection_cache)
        snapshot["detector_calls"] = stats["detector_runs"]
        snapshot.setdefault("decode_failures", decode_failures)
        return snapshot

    def debug_break(reason: str, frame_idx: int | None = None) -> None:
        if not debug:
            return
        print(f"[motion_debug] {reason} frame={frame_idx}")
        breakpoint()

    def decode_frame(idx: int) -> tuple[bool, np.ndarray | None]:
        nonlocal current_pos, frame_width, frame_height, decode_failures
        if idx < 0 or idx >= total:
            return False, None
        if current_pos is None or idx != current_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            stats["seeks"] += 1
        ret, frame = cap.read()
        if not ret:
            current_pos = None
            decode_failures += 1
            if decode_failures > decode_failure_limit:
                stats["decode_failures"] = decode_failures
                raise MotionWindowError(
                    f"Exceeded decode failure limit ({decode_failures}) while searching for the motion window",
                    stats=finalize_stats(),
                )
            return False, None
        decode_failures = 0
        current_pos = idx + 1
        stats["frames_decoded"] += 1
        if frame_width == 0 or frame_height == 0:
            frame_height, frame_width = frame.shape[:2]
        return True, frame

    def detect_frame(idx: int) -> dict | None:
        cached = detection_cache.get(idx, ...)
        if cached is not ...:
            return cached  # type: ignore[return-value]
        stats["frames_evaluated"] += 1
        if stats["frames_evaluated"] > max_eval_budget:
            raise MotionWindowError(
                "Motion window search exceeded the evaluation budget",
                stats=finalize_stats(),
            )
        ok, frame = decode_frame(idx)
        if not ok or frame is None:
            detection_cache[idx] = None
            return None
        enhanced, _ = preprocess_frame(frame)
        stats["detector_runs"] += 1
        detections = detector.detect(enhanced)
        best: dict | None = None
        best_score = float("-inf")
        for det in detections:
            score = det["score"]
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = det["bbox"]
            width = float(x2 - x1)
            height = float(y2 - y1)
            if width <= 0.0 or height <= 0.0:
                continue
            aspect = min(width, height) / max(width, height)
            if aspect < MOTION_WINDOW_MIN_ASPECT_RATIO:
                continue
            if frame_width and frame_height:
                if not bbox_within_image(det["bbox"], float(frame_width), float(frame_height)):
                    continue
            cx, cy, radius, _ = bbox_to_ball_metrics(x1, y1, x2, y2)
            if radius < MIN_BALL_RADIUS_PX:
                continue
            if score > best_score:
                center = np.array([cx, cy], dtype=float)
                best = {
                    "frame": idx,
                    "center": center,
                    "radius": float(radius),
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "score": float(score),
                }
                best_score = score
        detection_cache[idx] = best
        if best is not None:
            stats["frames_with_ball"] += 1
        return best

    try:
        if total == 0:
            stats["ball_never_observed"] = True
            raise MotionWindowNotFoundError(
                "Video contains no frames to analyse for ball motion",
                stats=finalize_stats(),
            )

        coarse_true: int | None = None
        coarse_false: int | None = None
        coarse_frames_total = 0
        coarse_steps = (64, 32, 16, 8, 4)
        for step in coarse_steps:
            idx = total - 1
            last_false = total
            frames_this_step = 0
            while idx >= 0:
                frames_this_step += 1
                det = detect_frame(idx)
                if det is not None:
                    coarse_true = idx
                    coarse_false = last_false if last_false < total else None
                    stats["coarse_step"] = step
                    stats["coarse_found_frame"] = idx
                    stats["coarse_false_frame"] = coarse_false
                    debug_break("coarse-hit", idx)
                    break
                last_false = idx
                idx -= step
            coarse_frames_total += frames_this_step
            if coarse_true is not None:
                break

        if coarse_true is None:
            stats["fallback_full_scan"] = True
            idx = total - 1
            frames_this_step = 0
            while idx >= 0:
                frames_this_step += 1
                det = detect_frame(idx)
                if det is not None:
                    coarse_true = idx
                    coarse_false = idx + 1 if idx + 1 < total else None
                    stats["coarse_step"] = 1
                    stats["coarse_found_frame"] = idx
                    stats["coarse_false_frame"] = coarse_false
                    debug_break("fallback-hit", idx)
                    break
                idx -= 1
            coarse_frames_total += frames_this_step

        stats["coarse_frames_scanned"] = coarse_frames_total

        if coarse_true is None:
            stats["ball_never_observed"] = True
            raise MotionWindowNotFoundError(
                "Unable to locate any ball detections in the video",
                stats=finalize_stats(),
            )

        initial_det = detect_frame(coarse_true)
        if initial_det is None:
            stats["ball_never_observed"] = True
            raise MotionWindowNotFoundError(
                "Initial coarse detection vanished before refinement",
                stats=finalize_stats(),
            )

        last_detection_idx = coarse_true
        last_center = initial_det["center"]
        refine_idx = coarse_true + 1
        misses = 0
        refine_frames = 0
        center_jump_limit = MAX_CENTER_JUMP_PX * 1.25

        while refine_idx < total and misses < miss_limit:
            det = detect_frame(refine_idx)
            refine_frames += 1
            if det is not None:
                center = det["center"]
                if last_center is None or np.linalg.norm(center - last_center) <= center_jump_limit:
                    last_detection_idx = refine_idx
                    last_center = center
                    misses = 0
                else:
                    misses += 1
            else:
                misses += 1
            refine_idx += 1

        stats["refine_frames"] = refine_frames
        stats["refine_last_frame"] = last_detection_idx
        if coarse_false is None and last_detection_idx >= total - 1:
            stats["ball_always_visible"] = True
            raise MotionWindowDegenerateError(
                "Ball detections persist through the final frame; unable to determine when the ball leaves the shot",
                stats=finalize_stats(),
            )

        start_frame = max(0, last_detection_idx - pad_frames)
        end_frame = min(total, last_detection_idx + confirm_radius + 1)
        if end_frame - start_frame > max_frames:
            start_frame = max(0, end_frame - max_frames)

        original_start = start_frame
        original_end = end_frame

        shift = MOTION_WINDOW_FORWARD_SHIFT
        if shift != 0:
            start_frame = min(total, start_frame + shift)
            end_frame = min(total, end_frame + shift)
            if end_frame - start_frame > max_frames:
                start_frame = max(0, end_frame - max_frames)
            if start_frame >= end_frame:
                start_frame = max(0, end_frame - 1)
        applied_shift = start_frame - original_start

        stats["frames_processed"] = len(detection_cache)
        stats["forward_shift"] = applied_shift
        stats["initial_window"] = (original_start, original_end)
        stats["detector_calls"] = stats["detector_runs"]
        stats["coarse_false_frame"] = coarse_false
        stats["last_detection_frame"] = last_detection_idx

        return start_frame, end_frame, True, stats
    finally:
        cap.release()


def process_video(
    video_path: str,
    ball_path: str,
    sticker_path: str,
    frames_dir: str = "ball_frames",
) -> str:
    """Process video_path saving ball trajectory and clubface center coordinates to JSON."""

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ball_compile_start = time.perf_counter()
    detector = TFLiteBallDetector("golf_ball_detector.tflite", conf_threshold=0.01)
    ball_compile_time = time.perf_counter() - ball_compile_start

    try:
        start_frame, end_frame, _, motion_stats = find_motion_window(
            video_path,
            detector,
            debug=MOTION_WINDOW_DEBUG,
        )
    except MotionWindowError as exc:
        raise RuntimeError(f"Motion window detection failed: {exc}") from exc
    processed = motion_stats.get("frames_processed", 0)
    detector_calls = motion_stats.get("detector_calls", 0)
    coarse_step = motion_stats.get("coarse_step")
    refine_frames = motion_stats.get("refine_frames")
    parts = [
        f"decoded {processed} frames",
        f"detector runs {detector_calls}",
    ]
    if isinstance(coarse_step, int) and coarse_step > 0:
        parts.append(f"coarse step {coarse_step}")
    if isinstance(refine_frames, int) and refine_frames > 0:
        parts.append(f"refine frames {refine_frames}")
    print(
        "Motion window frames: "
        f"{start_frame}-{end_frame} (" + ", ".join(parts) + ")"
    )

    clubface_tracker = ClubfaceCentroidTracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    inference_start = max(0, start_frame)
    inference_end = min(total_frames, end_frame)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
        for name in os.listdir(frames_dir):
            try:
                os.remove(os.path.join(frames_dir, name))
            except OSError:
                pass

    ball_time = 0.0
    clubface_time = 0.0
    ball_coords: list[dict] = []
    clubface_coords: list[dict] = []
    clubface_debug: list[dict] = []
    clubface_samples_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    processed_dot_frames: set[int] = set()
    prev_ir_gray: np.ndarray | None = None
    prev_ir_idx: int | None = None
    prev_ir_time: float | None = None
    prev_ir_mean: float | None = None
    prev_color: np.ndarray | None = None
    last_ball_center: np.ndarray | None = None
    last_ball_radius: float | None = None
    ball_velocity = np.zeros(2, dtype=float)
    last_ball_distance: float | None = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig = frame
        enhanced, _ = preprocess_frame(frame)
        if h is None:
            h, w = enhanced.shape[:2]
        ir_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        ir_gray = CLAHE.apply(ir_gray)
        if USE_BLUR:
            ir_gray = cv2.GaussianBlur(ir_gray, (3, 3), 0)
        t = frame_idx / video_fps
        in_window = inference_start <= frame_idx < inference_end

        detections: list[dict] = []
        if in_window:
            start = time.perf_counter()
            detections = detector.detect(enhanced)
            ball_time += time.perf_counter() - start

        detected = False
        if in_window and detections:
            edge_safe = [
                det
                for det in detections
                if bbox_within_image(det["bbox"], w, h)
            ]
            candidates = edge_safe if edge_safe else detections
            best_det = max(candidates, key=lambda d: d["score"])
            if best_det["score"] >= BALL_SCORE_THRESHOLD:
                x1, y1, x2, y2 = best_det["bbox"]
                cx, cy, rad, distance = bbox_to_ball_metrics(x1, y1, x2, y2)
                if rad >= MIN_BALL_RADIUS_PX:
                    center = np.array([cx, cy], dtype=float)
                    if last_ball_center is not None and np.linalg.norm(center - last_ball_center) > MAX_CENTER_JUMP_PX:
                        pass
                    else:
                        bx = (cx - w / 2.0) * distance / FOCAL_LENGTH
                        by = (cy - h / 2.0) * distance / FOCAL_LENGTH
                        bz = distance - 30.0
                        ball_coords.append(
                            {
                                "time": round(t, 3),
                                "x": round(bx, 2),
                                "y": round(by, 2),
                                "z": round(bz, 2),
                            }
                        )
                        last_ball_distance = distance
                        cv2.rectangle(
                            enhanced, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                        )
                        cv2.circle(enhanced, (int(cx), int(cy)), int(rad), (0, 255, 0), 2)
                        cv2.putText(
                            enhanced,
                            f"x:{bx:.2f} y:{by:.2f} z:{bz:.2f}",
                            (int(cx) + 10, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        if last_ball_center is not None:
                            ball_velocity = center - last_ball_center
                        last_ball_center = center
                        last_ball_radius = rad
                        detected = True

        if not detected and last_ball_center is not None and in_window:
            if last_ball_radius is not None:
                motion = ball_velocity
                if np.linalg.norm(motion) <= MAX_CENTER_JUMP_PX:
                    cx, cy = last_ball_center + motion
                    radius = last_ball_radius
                    cv2.circle(enhanced, (int(cx), int(cy)), int(radius), (0, 255, 255), 2)

        current_mean = float(ir_gray.mean())
        if prev_ir_gray is not None and prev_ir_idx is not None and prev_ir_time is not None:
            prev_mean = prev_ir_mean if prev_ir_mean is not None else float(prev_ir_gray.mean())
            on_color: np.ndarray | None = None
            off_color: np.ndarray | None = None
            if current_mean >= prev_mean:
                on_idx = frame_idx
                on_time = t
                on_gray = ir_gray
                off_gray = prev_ir_gray
                on_color = orig
                off_color = prev_color
            else:
                on_idx = prev_ir_idx
                on_time = prev_ir_time
                on_gray = prev_ir_gray
                off_gray = ir_gray
                on_color = prev_color
                off_color = orig
            if (
                on_idx not in processed_dot_frames
                and inference_start <= on_idx < inference_end
                and on_color is not None
                and off_color is not None
            ):
                start_clubface = time.perf_counter()
                dot_detections = detect_reflective_dots(off_color, on_color)
                observation, metrics = clubface_tracker.update(
                    dot_detections,
                    approx_depth_cm=last_ball_distance,
                )
                clubface_time += time.perf_counter() - start_clubface
                raw_center_metric = metrics.get("raw_center")
                if isinstance(raw_center_metric, np.ndarray):
                    metrics["raw_center"] = [
                        float(raw_center_metric[0]),
                        float(raw_center_metric[1]),
                    ]
                metrics["frame"] = int(on_idx)
                metrics["time"] = round(on_time, 3)
                metrics["distance_ref"] = last_ball_distance
                clubface_debug.append(metrics)
                if observation is not None:
                    processed_dot_frames.add(on_idx)
                    center_px = observation.get("center_px")
                    depth_cm = observation.get("depth_cm")
                    if depth_cm is not None and center_px is not None:
                        u = float(center_px[0])
                        v = float(center_px[1])
                        x_cm = (u - CX) * depth_cm / FX
                        y_cm = (v - CY) * depth_cm / FY
                        z_cm = depth_cm - CLUBFACE_Z_OFFSET_CM
                        time_rounded = round(on_time, 3)
                        clubface_coords.append(
                            {
                                "time": time_rounded,
                                "x": round(float(x_cm), 2),
                                "y": round(float(y_cm), 2),
                                "z": round(float(z_cm), 2),
                            }
                        )
                        detection_points = metrics.get("points")
                        clubface_samples_by_frame[on_idx].append(
                            {
                                "time": time_rounded,
                                "center_px": (u, v),
                                "points": detection_points if isinstance(detection_points, list) else [],
                            }
                        )
                        if on_idx == frame_idx:
                            cv2.putText(
                                enhanced,
                                f"X:{x_cm:.2f} Y:{y_cm:.2f} Z:{z_cm:.2f}",
                                (int(u) + 8, int(v) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 0, 0),
                                1,
                                cv2.LINE_AA,
                            )

        if frames_dir and inference_start <= frame_idx < inference_end:
            cv2.imwrite(
                os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), enhanced
            )

        prev_ir_gray = ir_gray
        prev_ir_idx = frame_idx
        prev_ir_time = t
        prev_ir_mean = current_mean
        prev_color = orig
        frame_idx += 1

    cap.release()

    ball_coords.sort(key=lambda c: c["time"])
    clubface_coords.sort(key=lambda c: c["time"])
    refined_trajectory: RefinedTrajectory | None = None
    if clubface_coords:
        refined_trajectory = refine_clubface_trajectory(clubface_coords)
        clubface_coords = refined_trajectory.coords
        if frames_dir:
            _annotate_clubface_frames(frames_dir, clubface_samples_by_frame, refined_trajectory)

    with open(ball_path, "w", encoding="utf-8") as f:
        json.dump(ball_coords, f, indent=2)
    with open(sticker_path, "w", encoding="utf-8") as f:
        json.dump(clubface_coords, f, indent=2)

    print(f"Saved {len(ball_coords)} ball points to {ball_path}")
    if clubface_coords:
        print(f"Saved {len(clubface_coords)} clubface points to {sticker_path}")
    else:
        print("Warning: No reflective dots detected; sticker file left empty")
    if clubface_debug:
        depth_ready = len(clubface_coords)
        print(
            f"Clubface frames processed: {len(clubface_debug)} | frames with depth: {depth_ready}"
        )
    print(f"Ball detection compile time: {ball_compile_time:.2f}s")
    print(f"Ball detection time: {ball_time:.2f}s")
    print(f"Clubface tracking time: {clubface_time:.2f}s")
    return "skibidi"



if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "degree_+10_2.mp4"
    ball_path = sys.argv[2] if len(sys.argv) > 2 else "ball_coords.json"
    sticker_path = sys.argv[3] if len(sys.argv) > 3 else "sticker_coords.json"
    frames_dir = sys.argv[4] if len(sys.argv) > 4 else "ball_frames"
    process_video(
        video_path,
        ball_path,
        sticker_path,
        frames_dir,
    )

