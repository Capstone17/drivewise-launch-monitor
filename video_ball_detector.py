import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict, deque
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
DOT_MAX_DETECTIONS = 6
DOT_MIN_Y_PX = 40.0
DOT_MIN_Y_FRACTION = 0.05
DOT_MIN_LEFT_COLUMN = 3
DOT_MIN_RIGHT_COLUMN = 1

CLUB_CANNY_LOW_THRESHOLD = int(round(0.21 * 255))
CLUB_CANNY_HIGH_THRESHOLD = int(round(0.55 * 255))
CLUB_CANNY_APERTURE_SIZE = 3
CLUB_RING_MIN_RADIUS = 7
CLUB_RING_MAX_RADIUS = 12
CLUB_RING_HALF_THICKNESS = 2
CLUB_RESPONSE_GAUSSIAN_SIZE = 5
CLUB_RESPONSE_GAUSSIAN_SIGMA = 1.0
CLUB_PRIMARY_THRESHOLD_SCALE = 0.45
CLUB_SECONDARY_THRESHOLD_SCALE = 0.35
CLUB_PEAK_MIN_SEPARATION_PX = 9
CLUB_CLUSTER_NEIGHBOR_RADIUS_PX = 28
CLUB_MAX_PEAK_CANDIDATES = 30
CLUB_RESIZE_WIDTH = 500
CLUB_RESIZE_HEIGHT = 500
CLUB_CLUSTER_SCORE_WEIGHT = 0.7
CLUB_CLUSTER_DENSITY_WEIGHT = 0.3

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
CLUBFACE_COLUMN_MAX_X_SPREAD_PX = 15.0
CLUBFACE_MIN_COLUMN_GAP_PX = 36.0
CLUBFACE_MIN_COLUMN_PAIR_GAP_PX = 24.0
CLUBFACE_MIN_HULL_AREA_PX = 2200.0
CLUBFACE_MIN_VERTICAL_SPAN_PX = 55.0
CLUBFACE_MAX_COLUMN_CURVE_RMSE_PX = 4.0
CLUBFACE_MAX_COLUMN_CURVE_ABS_PX = 6.5
CLUBFACE_TRAJECTORY_POLY_DEGREE = 2
CLUBFACE_LEFT_COLUMN_MAX_POINTS = 4
CLUBFACE_RIGHT_COLUMN_MAX_POINTS = 2
CLUBFACE_COLUMN_CLUSTER_MAX_ITER = 8
CLUB_TEMPLATE_MIN_AREA_PX = 1500
CLUB_TEMPLATE_UPDATE_MARGIN = 1.12
CLUB_TEMPLATE_RECOVERY_RATIO = 0.82
CLUB_TEMPLATE_MIN_EXTENSION_PX = 4.0
CLUB_TEMPLATE_MAX_EXTENSION_PX = 42.0
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

    @staticmethod
    def _fit_quadratic_column(points: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        count = points.shape[0]
        if count == 0:
            return np.ones(0, dtype=bool), 0.0, 0.0, 0.0
        if count <= 2:
            return np.ones(count, dtype=bool), 0.0, 0.0, float(CLUBFACE_MAX_COLUMN_CURVE_ABS_PX)
        y = points[:, 1]
        x = points[:, 0]
        y_spread = float(np.ptp(y)) if y.size else 0.0
        try:
            if y_spread < 1e-3:
                coeffs = np.array([0.0, 0.0, float(np.mean(x))], dtype=np.float64)
                predicted = np.full_like(x, coeffs[-1], dtype=np.float64)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", np.RankWarning)
                    coeffs = np.polyfit(y, x, 2)
                predicted = np.polyval(coeffs, y)
        except np.RankWarning:
            try:
                coeffs = np.polyfit(y, x, 1)
            except Exception:
                return np.zeros(count, dtype=bool), float("inf"), float("inf"), float("inf")
            predicted = np.polyval(coeffs, y)
        except Exception:
            return np.zeros(count, dtype=bool), float("inf"), float("inf"), float("inf")
        residuals = x - predicted
        abs_residuals = np.abs(residuals)
        median_residual = float(np.median(residuals)) if residuals.size else 0.0
        mad = float(np.median(np.abs(residuals - median_residual))) if residuals.size else 0.0
        robust_sigma = 1.4826 * mad if mad > 1e-6 else 0.0
        dynamic_thresh = max(
            CLUBFACE_MAX_COLUMN_CURVE_ABS_PX,
            3.5 * robust_sigma,
            8.0,
        )
        keep = abs_residuals <= dynamic_thresh
        min_keep = max(2, int(np.ceil(0.6 * count)))
        if keep.sum() < min_keep:
            sorted_indices = np.argsort(abs_residuals)
            keep = np.zeros(count, dtype=bool)
            keep[sorted_indices[:min_keep]] = True
            dynamic_thresh = max(
                dynamic_thresh,
                float(abs_residuals[sorted_indices[:min_keep]].max()),
            )
        kept_residuals = residuals[keep]
        kept_abs = abs_residuals[keep]
        if kept_residuals.size == 0:
            kept_residuals = residuals
            kept_abs = abs_residuals
        rmse = float(np.sqrt(np.mean(kept_residuals ** 2))) if kept_residuals.size else 0.0
        max_dev = float(kept_abs.max()) if kept_abs.size else 0.0
        return keep.astype(bool), rmse, max_dev, float(dynamic_thresh)

    def _pattern_metrics(self, points: np.ndarray) -> dict[str, object]:
        metrics: dict[str, object] = {
            'pattern_priority': 0,
            'pattern_score': 0,
            'left_count': 0,
            'right_count': 0,
            'has_horizontal': False,
            'vertical_left_pairs': 0,
            'vertical_right_pairs': 0,
            'best_alignment': None,
            'pattern_valid': False,
            'left_span_px': None,
            'right_span_px': None,
            'column_gap_px': None,
            'column_pair_gap_px': None,
            'hull_area_px': None,
            'vertical_span_px': None,
            'curve_left_threshold_px': None,
            'curve_right_threshold_px': None,
        }
        if points.size == 0:
            return metrics

        column_mask, labels, _ = self._cluster_two_columns(points)
        if column_mask.size == 0:
            metrics['curve_mask'] = []
            return metrics
        curve_mask = column_mask.copy()

        left_idx = np.where((labels == 0) & column_mask)[0]
        right_idx = np.where((labels == 1) & column_mask)[0]
        left_points = points[left_idx] if left_idx.size else np.empty((0, 2), dtype=np.float32)
        right_points = points[right_idx] if right_idx.size else np.empty((0, 2), dtype=np.float32)

        if left_points.size:
            _, left_rmse, left_max_dev, left_threshold = self._fit_quadratic_column(left_points)
        else:
            left_rmse = left_max_dev = left_threshold = 0.0
        if right_points.size:
            _, right_rmse, right_max_dev, right_threshold = self._fit_quadratic_column(right_points)
        else:
            right_rmse = right_max_dev = right_threshold = 0.0

        metrics['curve_left_rmse'] = float(left_rmse)
        metrics['curve_left_max_dev'] = float(left_max_dev)
        metrics['curve_right_rmse'] = float(right_rmse)
        metrics['curve_right_max_dev'] = float(right_max_dev)
        metrics['curve_left_threshold_px'] = float(left_threshold)
        metrics['curve_right_threshold_px'] = float(right_threshold)
        metrics['curve_left_ok'] = bool(left_points.size)
        metrics['curve_right_ok'] = bool(right_points.size)

        left_count = int(left_points.shape[0])
        right_count = int(right_points.shape[0])
        metrics['left_count'] = left_count
        metrics['right_count'] = right_count

        left_span = float(np.ptp(left_points[:, 0])) if left_points.shape[0] > 1 else 0.0
        right_span = float(np.ptp(right_points[:, 0])) if right_points.shape[0] > 1 else 0.0
        metrics['left_span_px'] = left_span if left_points.size else 0.0
        metrics['right_span_px'] = right_span if right_points.size else 0.0

        columns_present = left_count > 0 and right_count > 0
        column_gap = None
        column_pair_gap = None
        hull_area = None
        vertical_span = None
        if columns_present:
            left_max = float(left_points[:, 0].max())
            right_min = float(right_points[:, 0].min())
            column_gap = right_min - left_max
            metrics['column_gap_px'] = column_gap
            left_sorted = left_points[np.argsort(left_points[:, 1])]
            right_sorted = right_points[np.argsort(right_points[:, 1])]
            min_pair_gap = float('inf')
            for lp in left_sorted:
                y_diff = np.abs(right_sorted[:, 1] - lp[1])
                idx = int(np.argmin(y_diff))
                gap_val = float(right_sorted[idx, 0] - lp[0])
                if gap_val < min_pair_gap:
                    min_pair_gap = gap_val
            if min_pair_gap != float('inf'):
                column_pair_gap = min_pair_gap
                metrics['column_pair_gap_px'] = column_pair_gap
            all_points = np.vstack((left_points, right_points))
            if all_points.shape[0] >= 3:
                hull = cv2.convexHull(all_points.astype(np.float32))
                hull_area = float(cv2.contourArea(hull))
            elif all_points.size:
                hull_area = 0.0
            vertical_span = float(np.ptp(all_points[:, 1])) if all_points.shape[0] else 0.0
            metrics['hull_area_px'] = hull_area
            metrics['vertical_span_px'] = vertical_span

        metrics['pattern_score'] = min(left_count, 4) + min(right_count, 2)

        candidates = self._build_depth_candidates(left_points, right_points)
        horizontal_candidates = [c for c in candidates if str(c['type']).startswith('horizontal')]
        left_vertical = [int(c.get('steps', 0)) for c in candidates if str(c['type']).startswith('vertical_left')]
        right_vertical = [int(c.get('steps', 0)) for c in candidates if str(c['type']).startswith('vertical_right')]

        has_horizontal = bool(horizontal_candidates)
        metrics['has_horizontal'] = has_horizontal
        metrics['vertical_left_pairs'] = max(left_vertical) if left_vertical else 0
        metrics['vertical_right_pairs'] = max(right_vertical) if right_vertical else 0

        if horizontal_candidates:
            alignment_vals = [
                float(abs(c.get('vertical_delta', 0.0)))
                for c in horizontal_candidates
                if np.isfinite(c.get('vertical_delta', 0.0))
            ]
            if alignment_vals:
                metrics['best_alignment'] = float(min(alignment_vals))

        pattern_priority = 0
        if has_horizontal and left_count >= 1 and right_count >= 1:
            pattern_priority = 3
        elif left_count >= 2 and metrics['vertical_left_pairs'] > 0:
            pattern_priority = 2
        elif right_count >= 2 and metrics['vertical_right_pairs'] > 0:
            pattern_priority = 2
        elif left_count >= 2 or right_count >= 2:
            pattern_priority = 1

        metrics['pattern_priority'] = pattern_priority
        allowed_left = left_count in {2, 3, 4}
        allowed_right = right_count in {1, 2}
        span_ok = (
            (left_count == 0 or left_span <= CLUBFACE_COLUMN_MAX_X_SPREAD_PX)
            and (right_count == 0 or right_span <= CLUBFACE_COLUMN_MAX_X_SPREAD_PX)
        )
        gap_ok = column_gap is not None and column_gap >= CLUBFACE_MIN_COLUMN_GAP_PX
        pair_gap_ok = column_pair_gap is not None and column_pair_gap >= CLUBFACE_MIN_COLUMN_PAIR_GAP_PX
        area_ok = hull_area is not None and hull_area >= CLUBFACE_MIN_HULL_AREA_PX
        vertical_ok = vertical_span is not None and vertical_span >= CLUBFACE_MIN_VERTICAL_SPAN_PX
        metrics['pattern_valid'] = bool(
            columns_present
            and allowed_left
            and allowed_right
            and span_ok
            and gap_ok
            and pair_gap_ok
            and area_ok
            and vertical_ok
            and metrics['curve_left_ok']
            and metrics['curve_right_ok']
        )
        metrics['curve_mask'] = curve_mask.tolist() if curve_mask.size else []
        return metrics

    def _select_cluster_mask(
        self,
        points: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object], np.ndarray | None]:
        count_points = points.shape[0]
        if count_points == 0:
            return np.zeros(0, dtype=bool), {
                'pattern_priority': 0,
                'pattern_score': 0,
                'spread': 0.0,
                'count': 0,
                'total_weight': 0.0,
            }, None

        best_score: tuple | None = None
        best_mask: np.ndarray | None = None
        best_info: dict[str, object] = {}
        best_center: np.ndarray | None = None
        best_invalid_score: tuple | None = None
        best_invalid_mask: np.ndarray | None = None
        best_invalid_info: dict[str, object] = {}
        best_invalid_center: np.ndarray | None = None

        for mask_int in range(1, 1 << count_points):
            used_bits = mask_int.bit_count()
            if used_bits < self.min_dots:
                continue
            mask = np.array([(mask_int >> idx) & 1 for idx in range(count_points)], dtype=bool)
            subset_indices = np.flatnonzero(mask)
            subset_points = points[subset_indices]
            subset_weights = weights[subset_indices]
            if subset_points.size == 0:
                continue

            while True:
                pattern_info = self._pattern_metrics(subset_points)
                curve_mask_subset = pattern_info.get('curve_mask')
                if curve_mask_subset is None:
                    break
                curve_mask_subset = np.array(curve_mask_subset, dtype=bool)
                if curve_mask_subset.size == 0 or curve_mask_subset.all():
                    break
                if not curve_mask_subset.any():
                    subset_points = np.empty((0, 2), dtype=np.float32)
                    break
                subset_indices = subset_indices[curve_mask_subset]
                subset_points = subset_points[curve_mask_subset]
                subset_weights = subset_weights[curve_mask_subset]
                mask = np.zeros(count_points, dtype=bool)
                mask[subset_indices] = True
                if subset_points.shape[0] < self.min_dots:
                    subset_points = np.empty((0, 2), dtype=np.float32)
                    break

            if subset_points.size == 0:
                continue

            used = int(mask.sum())
            if used < self.min_dots:
                continue

            center_candidate = self._weighted_center(points, weights, mask)
            if center_candidate is None:
                continue

            spread = self._cluster_spread(points, mask, center_candidate)
            pattern_info = self._pattern_metrics(subset_points)
            pattern_valid = bool(pattern_info.get('pattern_valid'))
            priority = int(pattern_info['pattern_priority'])
            if spread > self.max_spread_px and priority == 0:
                continue
            if not pattern_valid:
                invalid_score = (
                    int(pattern_info.get('pattern_score', 0)),
                    used,
                    float(subset_weights.sum()),
                    -float(spread),
                )
                if best_invalid_score is None or invalid_score > best_invalid_score:
                    best_invalid_score = invalid_score
                    best_invalid_mask = mask
                    best_invalid_center = center_candidate.astype(np.float32, copy=True)
                    best_invalid_info = {
                        **pattern_info,
                        'spread': float(spread),
                        'count': used,
                        'total_weight': float(subset_weights.sum()),
                        'pattern_valid': False,
                    }
                continue

            total_weight = float(subset_weights.sum())
            pattern_score = int(pattern_info['pattern_score'])
            horizontal_bonus = 1 if pattern_info.get('has_horizontal') else 0
            alignment_val = pattern_info.get('best_alignment')
            alignment_term = -float(alignment_val) if alignment_val is not None else float('-inf')

            candidate_score = (
                priority,
                pattern_score,
                used,
                horizontal_bonus,
                alignment_term,
                total_weight,
                -float(spread),
            )
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_mask = mask
                best_center = center_candidate.astype(np.float32, copy=True)
                best_info = {
                    **pattern_info,
                    'spread': float(spread),
                    'count': used,
                    'total_weight': total_weight,
                    'pattern_valid': True,
                }

        if best_mask is None:
            if best_invalid_mask is not None:
                return best_invalid_mask, best_invalid_info, best_invalid_center
            fallback_mask = np.zeros(count_points, dtype=bool)
            fallback_info = {
                'pattern_priority': 0,
                'pattern_score': 0,
                'spread': 0.0,
                'count': 0,
                'total_weight': 0.0,
                'pattern_valid': False,
                'left_count': 0,
                'right_count': 0,
                'has_horizontal': False,
                'vertical_left_pairs': 0,
                'vertical_right_pairs': 0,
                'best_alignment': None,
            }
            return fallback_mask, fallback_info, None

        return best_mask, best_info, best_center

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
            'pattern_valid': False,
        }
        if not detections:
            self._register_miss()
            return None, metrics

        points = np.array([det.centroid for det in detections], dtype=np.float32)
        weights = np.array([
            max(det.area, 1.0) * max(det.brightness, 1.0) for det in detections
        ], dtype=np.float32)

        order = weights.argsort()[::-1]
        if order.size > self.max_candidates:
            order = order[: self.max_candidates]
        points = points[order]
        weights = weights[order]

        if points.shape[0] < self.min_dots:
            metrics['status'] = 'insufficient_dots'
            metrics['used_dots'] = int(points.shape[0])
            self._register_miss()
            return None, metrics

        mask, cluster_info, center = self._select_cluster_mask(points, weights)
        used = int(mask.sum())
        if used < self.min_dots:
            metrics['status'] = 'insufficient_dots'
            metrics['used_dots'] = used
            self._register_miss()
            return None, metrics
        if not bool(cluster_info.get('pattern_valid', False)):
            metrics.update({
                'status': 'pattern_rejected',
                'used_dots': used,
                'pattern_priority': int(cluster_info.get('pattern_priority', 0)),
                'pattern_score': int(cluster_info.get('pattern_score', 0)),
                'left_dots': int(cluster_info.get('left_count', 0)),
                'right_dots': int(cluster_info.get('right_count', 0)),
                'pattern_valid': False,
                'column_gap_px': (float(cluster_info['column_gap_px']) if cluster_info.get('column_gap_px') is not None else None),
                'column_pair_gap_px': (float(cluster_info['column_pair_gap_px']) if cluster_info.get('column_pair_gap_px') is not None else None),
                'hull_area_px': (float(cluster_info['hull_area_px']) if cluster_info.get('hull_area_px') is not None else None),
                'vertical_span_px': (float(cluster_info['vertical_span_px']) if cluster_info.get('vertical_span_px') is not None else None),
                'curve_left_rmse': (float(cluster_info['curve_left_rmse']) if cluster_info.get('curve_left_rmse') is not None else None),
                'curve_right_rmse': (float(cluster_info['curve_right_rmse']) if cluster_info.get('curve_right_rmse') is not None else None),
                'curve_left_max_dev': (float(cluster_info['curve_left_max_dev']) if cluster_info.get('curve_left_max_dev') is not None else None),
                'curve_right_max_dev': (float(cluster_info['curve_right_max_dev']) if cluster_info.get('curve_right_max_dev') is not None else None),
                'curve_left_threshold_px': (float(cluster_info['curve_left_threshold_px']) if cluster_info.get('curve_left_threshold_px') is not None else None),
                'curve_right_threshold_px': (float(cluster_info['curve_right_threshold_px']) if cluster_info.get('curve_right_threshold_px') is not None else None),
                'curve_left_ok': bool(cluster_info.get('curve_left_ok', False)),
                'curve_right_ok': bool(cluster_info.get('curve_right_ok', False)),
            })
            self._register_miss()
            return None, metrics
        if center is None:
            center = self._weighted_center(points, weights, mask)
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

        spread_val = float(cluster_info.get('spread', self._cluster_spread(points, mask, raw_center)))
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
            'pattern_priority': int(cluster_info.get('pattern_priority', 0)),
            'pattern_score': int(cluster_info.get('pattern_score', 0)),
            'left_dots': int(cluster_info.get('left_count', 0)),
            'right_dots': int(cluster_info.get('right_count', 0)),
            'pattern_valid': True,
            'column_gap_px': (float(cluster_info['column_gap_px']) if cluster_info.get('column_gap_px') is not None else None),
            'column_pair_gap_px': (float(cluster_info['column_pair_gap_px']) if cluster_info.get('column_pair_gap_px') is not None else None),
            'hull_area_px': (float(cluster_info['hull_area_px']) if cluster_info.get('hull_area_px') is not None else None),
            'vertical_span_px': (float(cluster_info['vertical_span_px']) if cluster_info.get('vertical_span_px') is not None else None),
            'curve_left_rmse': (float(cluster_info['curve_left_rmse']) if cluster_info.get('curve_left_rmse') is not None else None),
            'curve_right_rmse': (float(cluster_info['curve_right_rmse']) if cluster_info.get('curve_right_rmse') is not None else None),
            'curve_left_max_dev': (float(cluster_info['curve_left_max_dev']) if cluster_info.get('curve_left_max_dev') is not None else None),
            'curve_right_max_dev': (float(cluster_info['curve_right_max_dev']) if cluster_info.get('curve_right_max_dev') is not None else None),
            'curve_left_threshold_px': (float(cluster_info['curve_left_threshold_px']) if cluster_info.get('curve_left_threshold_px') is not None else None),
            'curve_right_threshold_px': (float(cluster_info['curve_right_threshold_px']) if cluster_info.get('curve_right_threshold_px') is not None else None),
            'curve_left_ok': bool(cluster_info.get('curve_left_ok', False)),
            'curve_right_ok': bool(cluster_info.get('curve_right_ok', False)),
        })

        if confidence < self.min_confidence:
            metrics['status'] = 'low_confidence'
            self._register_miss()
            return None, metrics

        self.filtered_center = filtered
        self.prev_center = raw_center.copy()
        self.miss_streak = 0

        geometry = self._geometry_from_points(points[mask], filtered, approx_depth_cm)
        metrics['depth_cm'] = geometry['depth_cm']
        metrics['depth_source'] = geometry['depth_source']
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

        column_mask, labels, _ = self._cluster_two_columns(points)
        left_points = points[np.where((labels == 0) & column_mask)[0]] if column_mask.size else np.empty((0, 2), dtype=np.float32)
        right_points = points[np.where((labels == 1) & column_mask)[0]] if column_mask.size else np.empty((0, 2), dtype=np.float32)

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

        center_px = filtered_center.copy()

        if left_points.size:
            center_px[1] = float(left_points[:, 1].mean())
        elif right_points.size:
            center_px[1] = float(right_points[:, 1].mean())

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
        }

    @staticmethod
    def _cluster_two_columns(
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = points.shape[0]
        labels = np.full(count, -1, dtype=np.int8)
        centers = np.zeros(2, dtype=np.float32)
        if count == 0:
            return np.zeros(0, dtype=bool), labels, centers
        if count == 1:
            labels[0] = 0
            centers[:] = float(points[0, 0])
            return np.ones(1, dtype=bool), labels, centers

        x_vals = points[:, 0].astype(np.float32)
        sorted_idx = np.argsort(x_vals)
        half = max(1, count // 2)
        initial_left = x_vals[sorted_idx[:half]].mean()
        initial_right = x_vals[sorted_idx[half:]].mean()
        if not np.isfinite(initial_left) or not np.isfinite(initial_right):
            initial_left = float(x_vals.min())
            initial_right = float(x_vals.max())
        centers = np.array([initial_left, initial_right], dtype=np.float32)
        if centers[0] == centers[1]:
            centers[0] -= 0.5
            centers[1] += 0.5

        for _ in range(CLUBFACE_COLUMN_CLUSTER_MAX_ITER):
            distances = np.abs(x_vals[:, None] - centers[None, :])
            new_labels = distances.argmin(axis=1).astype(np.int8)
            if not (new_labels == 0).any() or not (new_labels == 1).any():
                median_x = float(np.median(x_vals))
                new_labels = (x_vals >= median_x).astype(np.int8)
                if not (new_labels == 0).any() or not (new_labels == 1).any():
                    new_labels = np.zeros(count, dtype=np.int8)
                    new_labels[sorted_idx[-1]] = 1
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for column in (0, 1):
                column_mask = labels == column
                if column_mask.any():
                    centers[column] = float(x_vals[column_mask].mean())
        else:
            labels = distances.argmin(axis=1).astype(np.int8)

        if centers[0] > centers[1]:
            centers = centers[::-1]
            labels = 1 - labels

        mask = np.zeros(count, dtype=bool)
        max_points_per_column = (CLUBFACE_LEFT_COLUMN_MAX_POINTS, CLUBFACE_RIGHT_COLUMN_MAX_POINTS)
        for column in (0, 1):
            column_idx = np.where(labels == column)[0]
            if column_idx.size == 0:
                continue
            x_column = x_vals[column_idx]
            center = centers[column]
            deviations = np.abs(x_column - center)
            mad = float(np.median(deviations))
            if not np.isfinite(mad):
                mad = 0.0
            robust_scale = 1.4826 * mad if mad > 1e-3 else float(np.std(x_column))
            if not np.isfinite(robust_scale) or robust_scale < 1e-3:
                robust_scale = float(np.max(deviations))
            tolerance = max(
                3.0,
                min(CLUBFACE_COLUMN_MAX_X_SPREAD_PX, 3.0 * robust_scale + 2.0),
            )
            keep_local = deviations <= tolerance
            if not keep_local.any():
                keep_local = np.zeros(column_idx.size, dtype=bool)
                keep_local[np.argmin(deviations)] = True
            kept_idx = column_idx[keep_local]
            if kept_idx.size > max_points_per_column[column]:
                y_vals = points[kept_idx, 1]
                sorted_order = np.argsort(y_vals)
                sorted_idx_by_y = kept_idx[sorted_order]
                if max_points_per_column[column] == 1:
                    kept_idx = np.array([sorted_idx_by_y[len(sorted_idx_by_y) // 2]], dtype=int)
                else:
                    y_min = float(points[sorted_idx_by_y[0], 1])
                    y_max = float(points[sorted_idx_by_y[-1], 1])
                    targets = np.linspace(y_min, y_max, max_points_per_column[column])
                    remaining = sorted_idx_by_y.tolist()
                    selected: list[int] = []
                    for target in targets:
                        distances_y = [abs(float(points[idx, 1]) - target) for idx in remaining]
                        pick_pos = int(np.argmin(distances_y))
                        selected.append(remaining.pop(pick_pos))
                        if not remaining:
                            break
                    kept_idx = np.array(selected, dtype=int)
            mask[kept_idx] = True
        return mask, labels, centers

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

def _ring_kernel(radius: int, half_thickness: int) -> np.ndarray:
    outer_radius = float(radius + half_thickness)
    inner_radius = max(0.0, float(radius - half_thickness))
    size = int(2 * math.ceil(outer_radius) + 1)
    coords = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    distance_sq = xx * xx + yy * yy
    outer_mask = distance_sq <= outer_radius * outer_radius
    inner_mask = distance_sq <= inner_radius * inner_radius
    kernel = outer_mask.astype(np.float32) - inner_mask.astype(np.float32)
    kernel -= kernel.mean()
    norm = float(np.linalg.norm(kernel))
    if norm > 1e-6:
        kernel /= norm
    return kernel


def _normalize01(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    values = values.astype(np.float32, copy=False)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)



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

    resized = cv2.resize(on_bgr, (CLUB_RESIZE_WIDTH, CLUB_RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    scale_x = width / float(CLUB_RESIZE_WIDTH)
    scale_y = height / float(CLUB_RESIZE_HEIGHT)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        gray,
        CLUB_CANNY_LOW_THRESHOLD,
        CLUB_CANNY_HIGH_THRESHOLD,
        apertureSize=CLUB_CANNY_APERTURE_SIZE,
        L2gradient=True,
    )
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, dilate_kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
    edges_float = edges.astype(np.float32) / 255.0

    best_score = np.zeros_like(edges_float, dtype=np.float32)
    best_radius = np.zeros_like(edges_float, dtype=np.float32)
    for radius in range(CLUB_RING_MIN_RADIUS, CLUB_RING_MAX_RADIUS + 1):
        kernel = _ring_kernel(radius, CLUB_RING_HALF_THICKNESS)
        response = cv2.filter2D(edges_float, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        mask = response > best_score
        best_score[mask] = response[mask]
        best_radius[mask] = float(radius)

    if CLUB_RESPONSE_GAUSSIAN_SIZE > 1:
        response_map = cv2.GaussianBlur(
            best_score,
            (CLUB_RESPONSE_GAUSSIAN_SIZE, CLUB_RESPONSE_GAUSSIAN_SIZE),
            CLUB_RESPONSE_GAUSSIAN_SIGMA,
            borderType=cv2.BORDER_REPLICATE,
        )
    else:
        response_map = best_score

    max_score = float(response_map.max())
    if max_score <= 1e-6:
        return []

    candidate_coords: np.ndarray | None = None
    for scale in (CLUB_PRIMARY_THRESHOLD_SCALE, CLUB_SECONDARY_THRESHOLD_SCALE):
        threshold = scale * max_score
        coords = np.column_stack(np.where(response_map >= threshold))
        if coords.size:
            candidate_coords = coords
            break
    if candidate_coords is None or candidate_coords.size == 0:
        return []

    scores = response_map[candidate_coords[:, 0], candidate_coords[:, 1]]
    order = np.argsort(scores)[::-1]
    coords_sorted = candidate_coords[order]
    scores_sorted = scores[order]

    selected_xy: list[tuple[float, float]] = []
    selected_scores: list[float] = []
    for (y, x), score in zip(coords_sorted, scores_sorted):
        if len(selected_xy) >= CLUB_MAX_PEAK_CANDIDATES:
            break
        if selected_xy:
            distances = np.linalg.norm(np.array(selected_xy) - np.array([x, y]), axis=1)
            if np.any(distances < CLUB_PEAK_MIN_SEPARATION_PX):
                continue
        selected_xy.append((float(x), float(y)))
        selected_scores.append(float(score))

    if not selected_xy:
        return []

    coords_arr = np.array(selected_xy, dtype=np.float32)
    scores_arr = np.array(selected_scores, dtype=np.float32)

    if coords_arr.shape[0] > 1:
        diff = coords_arr[:, None, :] - coords_arr[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
    else:
        dist = np.zeros((1, 1), dtype=np.float32)
    neighbor_counts = (dist <= CLUB_CLUSTER_NEIGHBOR_RADIUS_PX).sum(axis=1)

    score_norm = _normalize01(scores_arr)
    density_norm = _normalize01(neighbor_counts.astype(np.float32))
    rank_score = (
        CLUB_CLUSTER_SCORE_WEIGHT * score_norm
        + CLUB_CLUSTER_DENSITY_WEIGHT * density_norm
    )

    rank_order = np.argsort(rank_score)[::-1]
    rank_order = rank_order[: min(DOT_MAX_DETECTIONS, rank_order.size)]

    height_thresh = min(float(height) * DOT_MIN_Y_FRACTION, DOT_MIN_Y_PX)
    if height < 200:
        height_thresh *= float(height) / 200.0
    height_thresh = max(0.0, height_thresh)
    detections: list[DotDetection] = []
    for idx in rank_order:
        x_res, y_res = coords_arr[idx]
        score = scores_arr[idx]
        rx = int(np.clip(round(x_res), 0, best_radius.shape[1] - 1))
        ry = int(np.clip(round(y_res), 0, best_radius.shape[0] - 1))
        radius_res = float(best_radius[ry, rx])
        if radius_res <= 0.0:
            continue
        cx = float(x_res * scale_x)
        cy = float(y_res * scale_y)
        if cy < height_thresh:
            continue
        radius = radius_res * 0.5 * (scale_x + scale_y)
        area = math.pi * (radius ** 2)
        brightness = float(score * 255.0)
        detections.append(
            DotDetection(
                centroid=np.array([cx, cy], dtype=np.float32),
                area=area,
                brightness=brightness,
                circularity=1.0,
            )
        )

    if len(detections) < DOT_MIN_LEFT_COLUMN + DOT_MIN_RIGHT_COLUMN:
        return []

    coords_for_columns = np.array([d.centroid for d in detections], dtype=np.float32)
    sorted_x_idx = np.argsort(coords_for_columns[:, 0])
    sorted_x = coords_for_columns[sorted_x_idx, 0]
    gaps = np.diff(sorted_x)
    if gaps.size == 0:
        return []
    max_gap = float(gaps.max())
    if max_gap < CLUBFACE_COLUMN_SPLIT_PX:
        return []
    split = int(np.argmax(gaps) + 1)
    left_count = split
    right_count = coords_for_columns.shape[0] - split
    if left_count < DOT_MIN_LEFT_COLUMN or right_count < DOT_MIN_RIGHT_COLUMN:
        return []

    detections.sort(key=lambda d: d.brightness, reverse=True)
    if len(detections) > DOT_MAX_DETECTIONS:
        detections = detections[:DOT_MAX_DETECTIONS]
    return detections


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

    degree_target = min(CLUBFACE_TRAJECTORY_POLY_DEGREE, max(1, times.size - 1))
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

    interpolated_points = [
        p
        for p in trajectory.pixel_points
        if not p.get("is_original", False)
    ]

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

        path_points: list[tuple[int, int]] = []
        for point in trajectory.pixel_points:
            u = point.get("u")
            v = point.get("v")
            if u is None or v is None:
                continue
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            if 0 <= u < width and 0 <= v < height:
                path_points.append((int(round(u)), int(round(v))))
        if len(path_points) >= 2:
            cv2.polylines(
                image,
                [np.array(path_points, dtype=np.int32)],
                isClosed=False,
                color=(255, 215, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        elif len(path_points) == 1:
            cv2.circle(image, path_points[0], 3, (255, 215, 0), -1, cv2.LINE_AA)

        for point in interpolated_points:
            u = point.get("u")
            v = point.get("v")
            if u is None or v is None:
                continue
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            if 0 <= u < width and 0 <= v < height:
                cv2.circle(
                    image,
                    (int(round(u)), int(round(v))),
                    4,
                    (0, 215, 255),
                    -1,
                    cv2.LINE_AA,
                )

        samples = samples_by_frame.get(frame_idx, [])
        dot_points: set[tuple[int, int]] = set()
        for sample in samples:
            coords = sample.get("dot_centroids")
            if not coords:
                continue
            for coord in coords:
                if coord is None or len(coord) < 2:
                    continue
                try:
                    du = float(coord[0])
                    dv = float(coord[1])
                except (TypeError, ValueError):
                    continue
                if not (np.isfinite(du) and np.isfinite(dv)):
                    continue
                if 0 <= du < width and 0 <= dv < height:
                    dot_points.add((int(round(du)), int(round(dv))))

        for sample in samples:
            center = sample.get("center_px")
            time_val = sample.get("time")
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

        for point in dot_points:
            cv2.circle(
                image,
                point,
                3,
                (255, 0, 255),
                -1,
                cv2.LINE_AA,
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


def blackout_green_pixels(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower, upper)
    if int(np.count_nonzero(green_mask)) == 0:
        return image
    result = image.copy()
    result[green_mask > 0] = 0
    return result


class MotionForegroundExtractor:
    """Build a background model and extract foreground masks around the club and ball."""

    def __init__(
        self,
        *,
        min_background_frames: int = 15,
        learning_rate: float = 0.02,
        min_component_area: int = 60,
        max_component_area: int = 150_000,
        history_length: int = 3,
        persistence_threshold: int = 2,
        median_kernel_size: int = 5,
    ) -> None:
        self.min_background_frames = max(1, int(min_background_frames))
        self.learning_rate = float(learning_rate)
        self.min_component_area = max(1, int(min_component_area))
        self.max_component_area = max(self.min_component_area, int(max_component_area))
        self.history_length = max(1, int(history_length))
        self.persistence_threshold = max(1, min(self.history_length, int(persistence_threshold)))
        kernel = max(1, int(median_kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        self.median_kernel_size = kernel
        self._accumulator: np.ndarray | None = None
        self._frames_seen = 0
        self._history: deque[np.ndarray] = deque(maxlen=self.history_length)
        self._prev_mask: np.ndarray | None = None
        self._prev_centroid: tuple[float, float] | None = None
        self._prev_motion_vec: tuple[float, float] | None = None
        self._template_contour: np.ndarray | None = None
        self._template_area: float = 0.0
        self._template_size: tuple[int, int] = (0, 0)

    def reset(self) -> None:
        self._accumulator = None
        self._frames_seen = 0
        self._history.clear()
        self._prev_mask = None
        self._prev_centroid = None
        self._prev_motion_vec = None
        self._template_contour = None
        self._template_area = 0.0
        self._template_size = (0, 0)

    @staticmethod
    def _compute_centroid(mask: np.ndarray | None) -> tuple[float, float] | None:
        if mask is None or mask.size == 0:
            return None
        moments = cv2.moments(mask, binaryImage=True)
        area = moments["m00"]
        if area <= 1e-6:
            return None
        cx = float(moments["m10"] / area)
        cy = float(moments["m01"] / area)
        return cx, cy

    @staticmethod
    def _extract_primary_contour(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None or mask.size == 0:
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _maybe_update_template(
        self,
        mask: np.ndarray,
        centroid: tuple[float, float],
        area: float,
    ) -> None:
        if area < CLUB_TEMPLATE_MIN_AREA_PX:
            return
        contour = self._extract_primary_contour(mask)
        if contour is None:
            return
        update_required = self._template_contour is None or area > self._template_area * CLUB_TEMPLATE_UPDATE_MARGIN
        if not update_required:
            return
        contour_f = contour.reshape(-1, 2).astype(np.float32)
        centered = contour_f - np.array([[centroid[0], centroid[1]]], dtype=np.float32)
        self._template_contour = centered
        self._template_area = float(area)
        x, y, w, h = cv2.boundingRect(contour)
        self._template_size = (int(w), int(h))

    def _render_template(
        self,
        frame_shape: tuple[int, int],
        centroid: tuple[float, float] | None,
    ) -> np.ndarray | None:
        if self._template_contour is None or centroid is None:
            return None
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        contour_shifted = np.round(
            self._template_contour + np.array([[centroid[0], centroid[1]]], dtype=np.float32)
        ).astype(np.int32)
        cv2.fillPoly(mask, [contour_shifted.reshape(-1, 1, 2)], 255)
        return mask

    def _augment_with_motion_template(
        self,
        base_mask: np.ndarray,
        centroid: tuple[float, float] | None,
        movement_vec: tuple[float, float] | None,
    ) -> np.ndarray:
        if self._template_contour is None or self._template_area <= 0.0:
            return base_mask

        frame_shape = base_mask.shape
        anchor = centroid or self._prev_centroid
        if anchor is None:
            return base_mask

        template_masks: list[np.ndarray] = []
        anchor_template = self._render_template(frame_shape, anchor)
        if anchor_template is not None:
            template_masks.append(anchor_template)

        if (
            movement_vec is not None
            and self._template_size != (0, 0)
            and self._template_area > 0.0
        ):
            current_area = float(np.count_nonzero(base_mask))
            ratio = current_area / float(self._template_area)
            if ratio < CLUB_TEMPLATE_RECOVERY_RATIO:
                dx, dy = movement_vec
                magnitude = math.hypot(dx, dy)
                if magnitude > 1e-3:
                    deficit = max(0.0, CLUB_TEMPLATE_RECOVERY_RATIO - ratio)
                    scale = deficit / CLUB_TEMPLATE_RECOVERY_RATIO
                    max_dim = float(max(self._template_size))
                    extent = scale * max_dim
                    extent = max(extent, CLUB_TEMPLATE_MIN_EXTENSION_PX)
                    extent = min(extent, CLUB_TEMPLATE_MAX_EXTENSION_PX)
                    if extent > 0.5:
                        ux = dx / magnitude
                        uy = dy / magnitude
                        lead_centroid = (
                            anchor[0] + ux * extent,
                            anchor[1] + uy * extent,
                        )
                        lead_template = self._render_template(frame_shape, lead_centroid)
                        if lead_template is not None:
                            template_masks.append(lead_template)

        if not template_masks:
            return base_mask

        template_union = template_masks[0].copy()
        for extra in template_masks[1:]:
            template_union = cv2.bitwise_or(template_union, extra)

        combined = cv2.bitwise_or(base_mask, template_union)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        return combined

    def _predict_from_template(self, frame_shape: tuple[int, int]) -> np.ndarray | None:
        if self._template_contour is None or self._prev_centroid is None:
            return None
        target = self._prev_centroid
        if self._prev_motion_vec is not None:
            target = (
                target[0] + self._prev_motion_vec[0],
                target[1] + self._prev_motion_vec[1],
            )
        predicted = self._render_template(frame_shape, target)
        if predicted is None:
            return None
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        predicted = cv2.morphologyEx(predicted, cv2.MORPH_CLOSE, kernel)
        return predicted

    def update_background(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        frame_f32 = frame.astype(np.float32)
        if self._accumulator is None:
            self._accumulator = frame_f32.copy()
        else:
            cv2.accumulateWeighted(frame_f32, self._accumulator, self.learning_rate)
        self._frames_seen += 1

    @property
    def ready(self) -> bool:
        return self._accumulator is not None and self._frames_seen >= self.min_background_frames

    def _build_support_mask(
        self,
        shape: tuple[int, int],
        ball_bbox: tuple[float, float, float, float] | None,
        dot_points: Sequence[tuple[float, float]] | None,
        club_center: tuple[float, float] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        support = np.zeros(shape, dtype=np.uint8)
        ball_mask = np.zeros(shape, dtype=np.uint8)
        if dot_points:
            for point in dot_points:
                try:
                    u = int(round(float(point[0])))
                    v = int(round(float(point[1])))
                except (TypeError, ValueError):
                    continue
                cv2.circle(support, (u, v), 12, 255, -1, lineType=cv2.LINE_AA)
        if club_center is not None:
            try:
                u = int(round(float(club_center[0])))
                v = int(round(float(club_center[1])))
            except (TypeError, ValueError):
                pass
            else:
                cv2.circle(support, (u, v), 18, 255, -1, lineType=cv2.LINE_AA)
        if ball_bbox is not None:
            x1, y1, x2, y2 = ball_bbox
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
            radius = int(round(max(x2 - x1, y2 - y1) * 0.65))
            radius = max(radius, 6)
            cv2.circle(ball_mask, (cx, cy), radius, 255, -1, lineType=cv2.LINE_AA)
        if np.count_nonzero(support):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            support = cv2.dilate(support, kernel, iterations=1)
        if np.count_nonzero(ball_mask):
            kernel_ball = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            ball_mask = cv2.dilate(ball_mask, kernel_ball, iterations=1)
        return support, ball_mask

    def _smooth_primary_region(
        self,
        mask: np.ndarray,
        brightness: np.ndarray | None,
    ) -> np.ndarray | None:
        if mask is None or mask.size == 0:
            return None
        binary = (mask > 0).astype(np.uint8)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=2,
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        core = None
        for radius in (10.0, 8.0, 6.0, 4.0):
            candidate = (dist >= radius).astype(np.uint8)
            if int(candidate.sum()) > 0:
                core = candidate
                break
        if core is None:
            core = binary.copy()

        core = cv2.morphologyEx(
            core,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        core = cv2.dilate(
            core,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            iterations=1,
        )

        soft = cv2.GaussianBlur(core.astype(np.float32), (0, 0), sigmaX=8.0, sigmaY=8.0)
        if soft.max() > 0:
            soft = cv2.normalize(soft, None, 0.0, 255.0, cv2.NORM_MINMAX)
        soft = soft.astype(np.uint8)
        _, refined = cv2.threshold(soft, 120, 255, cv2.THRESH_BINARY)
        refined = cv2.morphologyEx(
            refined,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        refined = cv2.GaussianBlur(refined, (9, 9), 0)
        _, refined = cv2.threshold(refined, 1, 255, cv2.THRESH_BINARY)

        dilated_original = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
            iterations=1,
        )
        refined = cv2.bitwise_and(refined, dilated_original)

        if brightness is not None and brightness.size:
            if not np.any((refined > 0) & brightness):
                return None
        return refined if int(np.count_nonzero(refined)) else None

    def make_mask(
        self,
        frame: np.ndarray,
        *,
        ball_bbox: tuple[float, float, float, float] | None = None,
        dot_points: Sequence[tuple[float, float]] | None = None,
        club_center: tuple[float, float] | None = None,
    ) -> np.ndarray | None:
        height, width = frame.shape[:2]
        support, ball_support = self._build_support_mask((height, width), ball_bbox, dot_points, club_center)
        support_pixels = int(np.count_nonzero(support))
        ball_pixels = int(np.count_nonzero(ball_support))
        if not self.ready:
            if support_pixels or ball_pixels:
                preliminary = cv2.bitwise_or(support, ball_support)
                if ball_pixels:
                    preliminary = cv2.bitwise_and(preliminary, cv2.bitwise_not(ball_support))
                return preliminary if int(np.count_nonzero(preliminary)) else None
            return None

        if self._accumulator is None or self._accumulator.shape != frame.shape:
            self.reset()
            if support_pixels or ball_pixels:
                preliminary = cv2.bitwise_or(support, ball_support)
                if ball_pixels:
                    preliminary = cv2.bitwise_and(preliminary, cv2.bitwise_not(ball_support))
                return preliminary if int(np.count_nonzero(preliminary)) else None
            return None

        background = cv2.convertScaleAbs(self._accumulator)
        diff = cv2.absdiff(frame, background)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        _, motion_mask = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if int(np.count_nonzero(motion_mask)) == 0:
            _, motion_mask = cv2.threshold(diff_gray, 18, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2,
        )
        if self.median_kernel_size > 1:
            motion_mask = cv2.medianBlur(motion_mask, self.median_kernel_size)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        mean_val, std_val = cv2.meanStdDev(gray_blur)
        mean_scalar = float(mean_val[0, 0]) if mean_val.size else 0.0
        std_scalar = float(std_val[0, 0]) if std_val.size else 0.0
        bright_thresh = mean_scalar + 1.5 * std_scalar
        bright_thresh = max(0.0, min(255.0, bright_thresh))
        if bright_thresh < 180.0:
            bright_thresh = 180.0
        _, bright_mask = cv2.threshold(gray_blur, bright_thresh, 255, cv2.THRESH_BINARY)
        bright_mask = cv2.dilate(
            bright_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        combined = cv2.bitwise_or(motion_mask, bright_mask)
        if support_pixels:
            combined = cv2.bitwise_or(combined, support)
        if ball_pixels:
            combined = cv2.bitwise_or(combined, ball_support)
        if int(np.count_nonzero(combined)) == 0:
            return None

        if self.history_length > 1:
            current_binary = (combined > 0).astype(np.uint8)
            if self._history and self._history[0].shape != current_binary.shape:
                self._history.clear()
            self._history.append(current_binary)
            accum = np.zeros_like(current_binary, dtype=np.uint16)
            for past in self._history:
                accum += past
            stable = np.where(accum >= self.persistence_threshold, 255, 0).astype(np.uint8)
        else:
            stable = combined

        bright_bool = bright_mask.astype(bool)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stable)
        filtered = np.zeros_like(stable)
        support_bool = support.astype(bool)
        support_has_pixels = bool(support_pixels)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_area or area > self.max_component_area:
                continue
            component_mask = labels == label
            if support_has_pixels and not np.any(component_mask & support_bool):
                continue
            if not np.any(component_mask & bright_bool):
                continue
            filtered[component_mask] = 255

        if int(np.count_nonzero(filtered)) == 0 and support_has_pixels:
            filtered = support.copy()
        else:
            filtered = cv2.bitwise_or(filtered, support)

        if ball_pixels:
            filtered = cv2.bitwise_and(filtered, cv2.bitwise_not(ball_support))

        filtered = cv2.morphologyEx(
            filtered,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
        filtered = cv2.dilate(
            filtered,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        nonzero = int(np.count_nonzero(filtered))
        if nonzero == 0:
            return None

        num_labels_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(filtered)
        if num_labels_final <= 1:
            return filtered
        areas_final = stats_final[1:, cv2.CC_STAT_AREA]
        if areas_final.size == 0:
            return None
        largest_label = 1 + int(np.argmax(areas_final))
        largest_mask = np.where(labels_final == largest_label, 255, 0).astype(np.uint8)
        if not np.any((labels_final == largest_label) & bright_bool):
            return None
        frame_shape = (height, width)
        smoothed = self._smooth_primary_region(largest_mask, bright_bool)
        if smoothed is None:
            fallback = self._predict_from_template(frame_shape)
            if fallback is not None:
                fallback_centroid = self._compute_centroid(fallback)
                prev_centroid = self._prev_centroid
                if fallback_centroid is not None and prev_centroid is not None:
                    self._prev_motion_vec = (
                        fallback_centroid[0] - prev_centroid[0],
                        fallback_centroid[1] - prev_centroid[1],
                    )
                elif fallback_centroid is not None:
                    self._prev_motion_vec = None
                self._prev_mask = fallback.copy()
                self._prev_centroid = fallback_centroid
                return fallback
            self._prev_mask = None
            self._prev_centroid = None
            self._prev_motion_vec = None
            return None

        area_est = float(np.count_nonzero(smoothed))
        centroid = self._compute_centroid(smoothed)
        movement_vec: tuple[float, float] | None = None
        if centroid is not None:
            if self._prev_centroid is not None:
                movement_vec = (
                    centroid[0] - self._prev_centroid[0],
                    centroid[1] - self._prev_centroid[1],
                )
            self._maybe_update_template(smoothed, centroid, area_est)
        elif self._prev_motion_vec is not None:
            movement_vec = self._prev_motion_vec

        enforced = self._augment_with_motion_template(smoothed, centroid, movement_vec)
        final_centroid = self._compute_centroid(enforced)
        if final_centroid is not None and self._prev_centroid is not None:
            self._prev_motion_vec = (
                final_centroid[0] - self._prev_centroid[0],
                final_centroid[1] - self._prev_centroid[1],
            )
        elif final_centroid is not None:
            self._prev_motion_vec = None
        else:
            self._prev_motion_vec = movement_vec

        self._prev_mask = enforced.copy()
        self._prev_centroid = final_centroid
        return enforced


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
        self.interpreter = Interpreter(model_path=model_path, num_threads=4)
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
    foreground_extractor = MotionForegroundExtractor()
    dot_points_by_frame: dict[int, list[tuple[float, float]]] = {}
    club_centers_by_frame: dict[int, tuple[float, float]] = {}

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
    prev_mask_ready = False
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
        base_frame = enhanced.copy()
        if frame_idx < inference_start or frame_idx >= inference_end:
            foreground_extractor.update_background(base_frame)
        if h is None:
            h, w = base_frame.shape[:2]
        ir_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        ir_gray = CLAHE.apply(ir_gray)
        if USE_BLUR:
            ir_gray = cv2.GaussianBlur(ir_gray, (3, 3), 0)
        t = frame_idx / video_fps
        in_window = inference_start <= frame_idx < inference_end

        dot_points_for_frame = dot_points_by_frame.pop(frame_idx, [])
        club_center_for_frame = club_centers_by_frame.pop(frame_idx, None)
        ball_overlay: dict[str, object] | None = None
        ball_prediction_overlay: dict[str, object] | None = None
        text_overlays: list[dict[str, object]] = []
        ball_bbox_for_mask: tuple[float, float, float, float] | None = None

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
                        ball_overlay = {
                            "center": (int(round(cx)), int(round(cy))),
                            "radius": int(round(rad)),
                        }
                        text_overlays.append(
                            {
                                "value": f"x:{bx:.2f} y:{by:.2f} z:{bz:.2f}",
                                "position": (int(round(cx)) + 10, int(round(cy))),
                                "color": (0, 255, 0),
                                "scale": 0.5,
                                "thickness": 1,
                            }
                        )
                        ball_bbox_for_mask = (x1, y1, x2, y2)
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
                    center_int = (int(round(cx)), int(round(cy)))
                    ball_prediction_overlay = {
                        "center": center_int,
                        "radius": int(round(radius)),
                    }
                    ball_bbox_for_mask = (
                        float(cx - radius),
                        float(cy - radius),
                        float(cx + radius),
                        float(cy + radius),
                    )

        mask = None
        masked_color_current = orig
        if in_window:
            mask = foreground_extractor.make_mask(
                base_frame,
                ball_bbox=ball_bbox_for_mask,
                dot_points=dot_points_for_frame,
                club_center=club_center_for_frame,
            )
        if mask is not None:
            masked_color_current = cv2.bitwise_and(orig, orig, mask=mask)
            output_frame = cv2.bitwise_and(base_frame, base_frame, mask=mask)
        else:
            output_frame = base_frame.copy()
        mask_ready_for_detection = mask is not None
        masked_color_current = blackout_green_pixels(masked_color_current)
        output_frame = blackout_green_pixels(output_frame)

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
                on_color = masked_color_current if mask_ready_for_detection else None
                off_color = prev_color if prev_mask_ready else None
            else:
                on_idx = prev_ir_idx
                on_time = prev_ir_time
                on_gray = prev_ir_gray
                off_gray = ir_gray
                on_color = prev_color if prev_mask_ready else None
                off_color = masked_color_current if mask_ready_for_detection else None
            if (
                on_idx not in processed_dot_frames
                and inference_start <= on_idx < inference_end
                and on_color is not None
                and off_color is not None
            ):
                start_clubface = time.perf_counter()
                dot_detections = detect_reflective_dots(off_color, on_color)
                dot_centroids: list[tuple[float, float]] = []
                for det in dot_detections:
                    centroid = getattr(det, "centroid", None)
                    if centroid is None or len(centroid) < 2:
                        continue
                    try:
                        du = float(centroid[0])
                        dv = float(centroid[1])
                    except (TypeError, ValueError):
                        continue
                    if not (np.isfinite(du) and np.isfinite(dv)):
                        continue
                    dot_centroids.append((du, dv))
                if dot_centroids and on_idx == frame_idx:
                    dot_points_for_frame = dot_centroids
                    dot_points_by_frame[frame_idx + 1] = dot_centroids
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
                time_rounded = round(on_time, 3)
                sample_entry: dict[str, object] = {"time": time_rounded}
                if dot_centroids:
                    sample_entry["dot_centroids"] = dot_centroids
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
                        clubface_coords.append(
                            {
                                "time": time_rounded,
                                "x": round(float(x_cm), 2),
                                "y": round(float(y_cm), 2),
                                "z": round(float(z_cm), 2),
                            }
                        )
                        sample_entry["center_px"] = (u, v)
                        clubface_samples_by_frame[on_idx].append(sample_entry)
                        if on_idx == frame_idx:
                            club_center_for_frame = (u, v)
                            club_centers_by_frame[frame_idx + 1] = (u, v)
                            text_overlays.append(
                                {
                                    "value": f"X:{x_cm:.2f} Y:{y_cm:.2f} Z:{z_cm:.2f}",
                                    "position": (int(round(u)) + 8, int(round(v)) - 8),
                                    "color": (255, 0, 0),
                                    "scale": 0.45,
                                    "thickness": 1,
                                }
                            )
                elif "dot_centroids" in sample_entry:
                    clubface_samples_by_frame[on_idx].append(sample_entry)

        if ball_overlay is not None:
            cx_i, cy_i = ball_overlay["center"]
            radius_i = max(int(ball_overlay["radius"]), 1)
            cv2.circle(output_frame, (cx_i, cy_i), radius_i + 3, (0, 0, 0), -1, cv2.LINE_AA)
        if ball_prediction_overlay is not None:
            center_pred = ball_prediction_overlay["center"]
            radius_pred = max(int(ball_prediction_overlay["radius"]), 1)
            cv2.circle(output_frame, center_pred, radius_pred + 3, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(output_frame, center_pred, radius_pred + 5, (0, 255, 255), 1, cv2.LINE_AA)
        for text_meta in text_overlays:
            cv2.putText(
                output_frame,
                text_meta["value"],
                (int(text_meta["position"][0]), int(text_meta["position"][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(text_meta.get("scale", 0.45)),
                text_meta.get("color", (255, 0, 0)),
                int(text_meta.get("thickness", 1)),
                cv2.LINE_AA,
            )

        if frames_dir and inference_start <= frame_idx < inference_end:
            cv2.imwrite(
                os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), output_frame
            )

        prev_ir_gray = ir_gray
        prev_ir_idx = frame_idx
        prev_ir_time = t
        prev_ir_mean = current_mean
        prev_color = masked_color_current
        prev_mask_ready = mask_ready_for_detection
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
    video_path = sys.argv[1] if len(sys.argv) > 1 else "CEsticker_white_200exp1.mp4"
    ball_path = sys.argv[2] if len(sys.argv) > 2 else "ball_coords.json"
    sticker_path = sys.argv[3] if len(sys.argv) > 3 else "sticker_coords.json"
    frames_dir = sys.argv[4] if len(sys.argv) > 4 else "ball_frames"
    process_video(
        video_path,
        ball_path,
        sticker_path,
        frames_dir,
    )
