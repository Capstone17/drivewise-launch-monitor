import json
import math
import os
import sys
import time
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
MAX_MOTION_FRAMES = 40  # maximum allowed motion window length in frames
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
USE_BLUR = False

# Reflective dot tracking parameters (mirroring MATLAB demo defaults)
DOT_MIN_AREA_PX = 15
DOT_MAX_AREA_PX = 400
DOT_MIN_BRIGHTNESS = 60.0
DOT_MIN_CIRCULARITY = 0.25
DOT_MAX_DETECTIONS = 4
TOPHAT_RADIUS_PX = 12
ADAPTIVE_SENSITIVITY = 0.38
STATIC_THRESHOLD = 0.30
WHITE_VALUE_THRESHOLD = 0.88
WHITE_SAT_MAX = 0.25
ADAPTIVE_BLOCK_SIZE = 35  # odd kernel size for adaptive thresholding

TOPHAT_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (2 * TOPHAT_RADIUS_PX + 1, 2 * TOPHAT_RADIUS_PX + 1)
)
OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

CLUBFACE_MAX_SPREAD_PX = 140.0
CLUBFACE_MAX_JUMP_PX = 80.0
CLUBFACE_EMA_ALPHA = 0.4
CLUBFACE_MIN_DOTS = 1
CLUBFACE_MAX_MISSES = 8
CLUBFACE_MAX_CANDIDATES = 6
CLUBFACE_MIN_CONFIDENCE = 0.35

# Pattern geometry: three markers stacked vertically on the heel side
# (2 cm separation) and a single marker 6.5 cm toward the toe, roughly
# aligned with the middle dot. We treat the clubface center as midway
# between the heel column and the toe marker.
CLUBFACE_VERTICAL_SPACING_CM = 2.0
CLUBFACE_HORIZONTAL_SPACING_CM = 6.5
CLUBFACE_COLUMN_SPLIT_PX = 10.0
CLUBFACE_DEPTH_MIN_CM = 10.0
CLUBFACE_DEPTH_MAX_CM = 400.0
CLUBFACE_CENTER_OFFSET_CM = CLUBFACE_HORIZONTAL_SPACING_CM / 2.0
CLUBFACE_Z_OFFSET_CM = 30.0


@dataclass
class DotDetection:
    centroid: np.ndarray
    area: float
    brightness: float
    circularity: float




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

        left_idx, right_idx = self._split_columns(points)
        left_points = points[left_idx] if left_idx.size else np.empty((0, 2), dtype=np.float32)
        right_points = points[right_idx] if right_idx.size else np.empty((0, 2), dtype=np.float32)

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
                            'type': f'vertical_{steps}',
                            'points': (ordered[i], ordered[j]),
                            'steps': steps,
                        }
                    )
        if left_points.shape[0] >= 1 and right_points.shape[0] >= 1:
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
                        'type': 'horizontal',
                        'points': (lp, right_point.copy()),
                        'steps': 0,
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
            if ctype == 'horizontal':
                return 0
            if ctype == 'vertical_1':
                return 1
            return 2

        ref = approx_depth_cm if approx_depth_cm is not None else self.last_depth_cm
        filtered = [c for c in candidates if CLUBFACE_DEPTH_MIN_CM <= c['depth_cm'] <= CLUBFACE_DEPTH_MAX_CM]
        pool = filtered if filtered else candidates
        if ref is not None:
            pool.sort(key=lambda c: (priority(c), abs(float(c['depth_cm']) - ref)))
        else:
            pool.sort(key=lambda c: (priority(c), -float(c['pixels'])))
        return pool[0] if pool else None
def _weighted_centroid(intensity_roi: np.ndarray, mask: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    y_idx, x_idx = np.nonzero(mask)
    if x_idx.size == 0:
        return np.array(offset, dtype=np.float32)
    weights = intensity_roi[y_idx, x_idx].astype(np.float32)
    total = float(weights.sum())
    if total <= 1e-6:
        return np.array(offset, dtype=np.float32)
    x = (x_idx.astype(np.float32) * weights).sum() / total
    y = (y_idx.astype(np.float32) * weights).sum() / total
    return np.array([offset[0] + x, offset[1] + y], dtype=np.float32)


def detect_reflective_dots(
    off_frame: np.ndarray | None, on_frame: np.ndarray | None
) -> list[DotDetection]:
    if on_frame is None or on_frame.size == 0:
        return []

    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2 or image.shape[-1] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def _invert(image: np.ndarray | None) -> np.ndarray | None:
        if image is None:
            return None
        return cv2.bitwise_not(image)

    on_frame = _invert(on_frame)
    off_frame = _invert(off_frame)

    on_bgr = _ensure_bgr(on_frame)
    gray = cv2.cvtColor(on_bgr, cv2.COLOR_BGR2GRAY)
    gray_clahe = CLAHE.apply(gray)

    tophat = cv2.morphologyEx(gray_clahe, cv2.MORPH_TOPHAT, TOPHAT_KERNEL)
    tophat_norm = cv2.normalize(
        tophat.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX
    )

    block_size = ADAPTIVE_BLOCK_SIZE if ADAPTIVE_BLOCK_SIZE % 2 == 1 else ADAPTIVE_BLOCK_SIZE + 1
    block_size = max(3, block_size)
    adaptive_c = max(0, int(round((1.0 - ADAPTIVE_SENSITIVITY) * 15)))
    adaptive_src = (tophat_norm * 255.0).astype(np.uint8)
    mask_adaptive = cv2.adaptiveThreshold(
        adaptive_src,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        adaptive_c,
    )

    static_mask = (tophat_norm >= STATIC_THRESHOLD).astype(np.uint8) * 255
    hsv = cv2.cvtColor(on_bgr, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2].astype(np.float32) / 255.0
    saturation_channel = hsv[:, :, 1].astype(np.float32) / 255.0
    mask_white = (
        (value_channel >= WHITE_VALUE_THRESHOLD)
        & (saturation_channel <= WHITE_SAT_MAX)
    )

    mask = cv2.bitwise_and(mask_adaptive, static_mask)
    mask = cv2.bitwise_and(mask, (mask_white.astype(np.uint8) * 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    detections: list[DotDetection] = []
    gray_weights = gray_clahe.astype(np.float32)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < DOT_MIN_AREA_PX or area > DOT_MAX_AREA_PX:
            continue
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        roi_mask = labels[y : y + h, x : x + w] == label
        if not np.any(roi_mask):
            continue
        roi_intensity = gray_weights[y : y + h, x : x + w]
        brightness = float(roi_intensity[roi_mask].mean())
        if brightness < DOT_MIN_BRIGHTNESS:
            continue
        contour_img = (roi_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(
            contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        perimeter = float(cv2.arcLength(contours[0], True))
        if perimeter <= 1e-6:
            continue
        circularity = float((4.0 * math.pi * area) / (perimeter * perimeter))
        if circularity < DOT_MIN_CIRCULARITY:
            continue
        centroid = _weighted_centroid(roi_intensity, roi_mask, (x, y))
        detections.append(
            DotDetection(
                centroid=centroid,
                area=float(area),
                brightness=brightness,
                circularity=circularity,
            )
        )

    detections.sort(key=lambda d: d.brightness, reverse=True)
    if len(detections) > DOT_MAX_DETECTIONS:
        detections = detections[:DOT_MAX_DETECTIONS]

    return detections




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
    }

    detection_cache: dict[int, dict | None] = {}
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    current_pos: int | None = None

    def debug_break(reason: str, frame_idx: int | None = None) -> None:
        if not debug:
            return
        print(f"[motion_debug] {reason} frame={frame_idx}")
        breakpoint()

    def decode_frame(idx: int) -> tuple[bool, np.ndarray | None]:
        nonlocal current_pos, frame_width, frame_height
        if idx < 0 or idx >= total:
            return False, None
        if current_pos is None or idx != current_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            stats["seeks"] += 1
        ret, frame = cap.read()
        if not ret:
            current_pos = None
            return False, None
        current_pos = idx + 1
        stats["frames_decoded"] += 1
        if frame_width == 0 or frame_height == 0:
            frame_height, frame_width = frame.shape[:2]
        return True, frame

    def detect_frame(idx: int) -> dict | None:
        stats["frames_evaluated"] += 1
        cached = detection_cache.get(idx, ...)
        if cached is not ...:
            return cached  # type: ignore[return-value]
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
            stats["frames_processed"] = 0
            stats["detector_calls"] = 0
            return 0, 0, False, stats

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
            stats["frames_processed"] = len(detection_cache)
            stats["detector_calls"] = stats["detector_runs"]
            return 0, total, False, stats

        initial_det = detect_frame(coarse_true)
        if initial_det is None:
            stats["frames_processed"] = len(detection_cache)
            stats["detector_calls"] = stats["detector_runs"]
            return 0, total, False, stats

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

        start_frame = max(0, last_detection_idx - pad_frames)
        end_frame = min(total, last_detection_idx + confirm_radius + 1)
        if end_frame - start_frame > max_frames:
            start_frame = max(0, end_frame - max_frames)

        stats["frames_processed"] = len(detection_cache)
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

    start_frame, end_frame, ball_found, motion_stats = find_motion_window(
        video_path,
        detector,
        debug=MOTION_WINDOW_DEBUG,
    )
    if not ball_found:
        raise RuntimeError("No ball detected in the video")
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
                        clubface_coords.append(
                            {
                                "time": round(on_time, 3),
                                "x": round(float(x_cm), 2),
                                "y": round(float(y_cm), 2),
                                "z": round(float(z_cm), 2),
                            }
                        )
                        if on_idx == frame_idx:
                            center_pt = (int(round(u)), int(round(v)))
                            cv2.circle(enhanced, center_pt, 6, (255, 0, 0), 2)
                            cv2.putText(
                                enhanced,
                                f"X:{x_cm:.2f} Y:{y_cm:.2f} Z:{z_cm:.2f}",
                                (center_pt[0] + 8, center_pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 0, 0),
                                1,
                                cv2.LINE_AA,
                            )
                    if on_idx == frame_idx:
                        raw_center = observation.get("raw_center")
                        if isinstance(raw_center, np.ndarray):
                            raw_pt = (
                                int(round(float(raw_center[0]))),
                                int(round(float(raw_center[1]))),
                            )
                            cv2.circle(enhanced, raw_pt, 3, (0, 0, 255), -1)

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
    video_path = sys.argv[1] if len(sys.argv) > 1 else "black_swing_1.mp4"
    ball_path = sys.argv[2] if len(sys.argv) > 2 else "ball_coords.json"
    sticker_path = sys.argv[3] if len(sys.argv) > 3 else "sticker_coords.json"
    frames_dir = sys.argv[4] if len(sys.argv) > 4 else "ball_frames"
    process_video(
        video_path,
        ball_path,
        sticker_path,
        frames_dir,
    )
