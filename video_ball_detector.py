import json
import itertools
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# Ensure Ultralytics can write its settings locally and skip auto-installation
os.environ.setdefault(
    "YOLO_CONFIG_DIR", os.path.join(os.path.dirname(__file__), ".yolo")
)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover - SciPy is optional
    linear_sum_assignment = None

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
BALL_SCORE_THRESHOLD = 0.4
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

MAX_MISSING_FRAMES = 12
MAX_MOTION_FRAMES = 40  # maximum allowed motion window length in frames
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
USE_BLUR = False

# Reflective dot tracking parameters
DOT_MIN_AREA_PX = 10
DOT_MAX_AREA_PX = 1600
DOT_MIN_BRIGHTNESS = 140.0
DOT_MIN_CIRCULARITY = 0.55
DOT_KERNEL_SIZE = 3
DOT_OPEN_ITER = 1
DOT_CLOSE_ITER = 1
DOT_ASSIGNMENT_MAX_DISTANCE = 32.0  # pixels
DOT_INLIER_THRESHOLD = 16.0  # pixels
DOT_RANSAC_REPROJECTION_ERROR = 6.0  # pixels
DOT_INITIAL_POSE_TRIALS = 300
CLUBFACE_AXIS_LENGTH = 40.0  # mm for debug axes


def _load_club_model(model_path: str) -> tuple[np.ndarray, np.ndarray]:
    default_points = np.array(
        [
            [-28.0, -18.5, 0.0],
            [30.0, -17.2, 0.0],
            [-25.4, 14.1, 0.0],
            [24.6, 15.3, 0.0],
            [-6.5, 28.8, 1.5],
            [8.4, 27.9, 1.3],
        ],
        dtype=np.float32,
    )
    default_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if not os.path.exists(model_path):
        return default_points, default_center
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return default_points, default_center
    dots = data.get("dots")
    center = data.get("clubface_center", [0.0, 0.0, 0.0])
    if not isinstance(dots, list) or len(dots) < 4:
        return default_points, default_center
    points = np.asarray(dots, dtype=np.float32)
    center_arr = np.asarray(center, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        return default_points, default_center
    if center_arr.shape != (3,):
        center_arr = default_center
    return points, center_arr


MODEL_POINTS, CLUBFACE_CENTER = _load_club_model(
    os.path.join(os.path.dirname(__file__), "club_model.json")
)


@dataclass
class DotDetection:
    centroid: np.ndarray
    area: float
    brightness: float
    circularity: float


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


def detect_reflective_dots(ir_off_gray: np.ndarray, ir_on_gray: np.ndarray) -> list[DotDetection]:
    diff = cv2.absdiff(ir_on_gray, ir_off_gray)
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DOT_KERNEL_SIZE, DOT_KERNEL_SIZE))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=DOT_OPEN_ITER)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=DOT_CLOSE_ITER)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    detections: list[DotDetection] = []
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        if area < DOT_MIN_AREA_PX or area > DOT_MAX_AREA_PX:
            continue
        roi_mask = labels[y : y + h, x : x + w] == label
        roi_intensity = ir_on_gray[y : y + h, x : x + w]
        masked_pixels = roi_intensity[roi_mask]
        if masked_pixels.size == 0:
            continue
        brightness = float(masked_pixels.mean())
        if brightness < DOT_MIN_BRIGHTNESS:
            continue
        contour_img = (roi_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    return detections


def _hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        return rows.astype(np.int32), cols.astype(np.int32)
    rows = cost.shape[0]
    cols = cost.shape[1]
    if rows == 0 or cols == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    if rows > cols:
        raise RuntimeError(
            "Hungarian fallback requires rows <= cols; reduce detections or install SciPy."
        )
    best_cost = float("inf")
    best_perm: Optional[Tuple[int, ...]] = None
    all_cols = range(cols)
    for perm in itertools.permutations(all_cols, rows):
        total = 0.0
        for r, c in enumerate(perm):
            total += float(cost[r, c])
            if total >= best_cost:
                break
        if total < best_cost:
            best_cost = total
            best_perm = perm
    if best_perm is None:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    rows_idx = np.arange(rows, dtype=np.int32)
    cols_idx = np.array(best_perm, dtype=np.int32)
    return rows_idx, cols_idx


class ReflectiveDotTracker:
    def __init__(
        self,
        model_points: np.ndarray,
        clubface_center: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        if model_points.shape[0] < 4:
            raise ValueError("At least four model points are required for pose estimation")
        self.model_points = model_points.astype(np.float32)
        self.clubface_center = clubface_center.astype(np.float32)
        self.camera_matrix = camera_matrix.astype(np.float32)
        self.dist_coeffs = dist_coeffs.astype(np.float32)
        self.prev_pose: Optional[tuple[np.ndarray, np.ndarray]] = None

    def project_model(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        projected, _ = cv2.projectPoints(self.model_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        return projected.reshape(-1, 2).astype(np.float32)

    def assign_matches(
        self,
        detections: Sequence[DotDetection],
        projected_points: np.ndarray,
    ) -> list[tuple[int, int, float]]:
        if not detections:
            return []
        max_assignable = min(len(detections), projected_points.shape[0])
        if max_assignable == 0:
            return []
        if len(detections) > max_assignable:
            order = np.argsort([-det.brightness for det in detections])[:max_assignable]
        else:
            order = np.arange(len(detections))
        det_points = np.array([detections[idx].centroid for idx in order], dtype=np.float32)
        diff = det_points[:, None, :] - projected_points[None, :, :]
        cost = np.linalg.norm(diff, axis=2)
        try:
            rows, cols = _hungarian(cost)
        except RuntimeError:
            return []
        matches: list[tuple[int, int, float]] = []
        for det_idx_local, model_idx in zip(rows, cols):
            distance = float(cost[det_idx_local, model_idx])
            if distance <= DOT_ASSIGNMENT_MAX_DISTANCE:
                matches.append((int(order[det_idx_local]), int(model_idx), distance))
        return matches

    def _initial_pose(
        self,
        detections: Sequence[DotDetection],
    ) -> Optional[tuple[np.ndarray, np.ndarray, list[tuple[int, int, float]]]]:
        if len(detections) < 4:
            return None
        det_points = np.array([det.centroid for det in detections], dtype=np.float32)
        best_pose: Optional[tuple[np.ndarray, np.ndarray]] = None
        best_matches: list[tuple[int, int, float]] = []
        best_inliers = 0
        model_indices = list(range(self.model_points.shape[0]))
        det_indices = list(range(len(detections)))
        trials = min(DOT_INITIAL_POSE_TRIALS, math.comb(len(det_indices), 4) * math.comb(len(model_indices), 4))
        for _ in range(trials):
            det_sample = random.sample(det_indices, 4)
            model_sample = random.sample(model_indices, 4)
            perm_list = list(itertools.permutations(model_sample))
            random.shuffle(perm_list)
            for perm in perm_list[: min(6, len(perm_list))]:
                object_pts = self.model_points[list(perm)].astype(np.float32)
                image_pts = det_points[det_sample].astype(np.float32)
                success, rvec, tvec = cv2.solvePnP(
                    object_pts,
                    image_pts,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_EPNP,
                )
                if not success:
                    continue
                projected = self.project_model(rvec, tvec)
                matches = self.assign_matches(detections, projected)
                inliers = [m for m in matches if m[2] <= DOT_INLIER_THRESHOLD]
                if len(inliers) > best_inliers or (
                    len(inliers) == best_inliers
                    and matches
                    and sum(m[2] for m in matches) < sum(m[2] for m in best_matches)
                ):
                    best_pose = (rvec, tvec)
                    best_matches = matches
                    best_inliers = len(inliers)
                if best_inliers >= 4:
                    break
            if best_inliers >= 4:
                break
        if best_pose is None or len(best_matches) < 4:
            return None
        return best_pose[0], best_pose[1], best_matches

    def estimate_pose(
        self,
        detections: Sequence[DotDetection],
    ) -> Optional[dict]:
        if len(detections) < 3:
            return None
        detection_points = np.array([det.centroid for det in detections], dtype=np.float32)
        candidate_matches: list[tuple[int, int, float]] = []
        if self.prev_pose is not None:
            rvec_prev, tvec_prev = self.prev_pose
            projected_prev = self.project_model(rvec_prev, tvec_prev)
            matches_prev = self.assign_matches(detections, projected_prev)
            inliers_prev = [m for m in matches_prev if m[2] <= DOT_INLIER_THRESHOLD]
            if len(inliers_prev) >= 4:
                candidate_matches = matches_prev
        if not candidate_matches:
            initial = self._initial_pose(detections)
            if initial is None:
                return None
            candidate_matches = initial[2]
        if len(candidate_matches) < 4:
            return None
        object_points = np.array(
            [self.model_points[m_idx] for _, m_idx, _ in candidate_matches], dtype=np.float32
        )
        image_points = np.array(
            [detection_points[d_idx] for d_idx, _, _ in candidate_matches], dtype=np.float32
        )
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            iterationsCount=100,
            reprojectionError=DOT_RANSAC_REPROJECTION_ERROR,
            confidence=0.99,
            flags=cv2.SOLVEPNP_AP3P,
        )
        if not success:
            return None
        inlier_mask = np.zeros(object_points.shape[0], dtype=bool)
        if inliers is not None and inliers.size:
            inlier_mask[inliers.flatten()] = True
        inlier_count = int(inlier_mask.sum()) if inliers is not None else object_points.shape[0]
        if inlier_count < 3:
            return None
        if inlier_count >= 4:
            refined_obj = object_points[inlier_mask]
            refined_img = image_points[inlier_mask]
            success_refine, rvec, tvec = cv2.solvePnP(
                refined_obj,
                refined_img,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success_refine:
                return None
        projected_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        residuals = np.linalg.norm(
            image_points - projected_points.reshape(-1, 2),
            axis=1,
        )
        if inliers is not None and inliers.size:
            reprojection_error = float(residuals[inliers.flatten()].mean())
        else:
            reprojection_error = float(residuals.mean())
        clubface_px, _ = cv2.projectPoints(
            self.clubface_center.reshape(1, 3),
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        self.prev_pose = (rvec.copy(), tvec.copy())
        return {
            "rvec": rvec.reshape(3, 1).astype(np.float32),
            "tvec": tvec.reshape(3, 1).astype(np.float32),
            "clubface_px": clubface_px.reshape(2).astype(np.float32),
            "inliers": inlier_count,
            "reprojection_error": reprojection_error,
        }


def rvec_to_euler(rvec: np.ndarray) -> tuple[float, float, float]:
    """Convert a rotation vector to roll, pitch and yaw in degrees."""
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    rotation_matrix = flip @ rotation_matrix
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0
    return tuple(np.degrees([roll, pitch, yaw]))


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
    """Process video_path saving ball and clubface pose information to JSON."""

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

    tracker_compile_start = time.perf_counter()
    dot_tracker = ReflectiveDotTracker(MODEL_POINTS, CLUBFACE_CENTER, CAMERA_MATRIX, DIST_COEFFS)
    tracker_compile_time = time.perf_counter() - tracker_compile_start

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
    dot_time = 0.0
    ball_coords: list[dict] = []
    clubface_records: list[dict] = []
    processed_dot_frames: set[int] = set()
    prev_ir_gray: np.ndarray | None = None
    prev_ir_idx: int | None = None
    prev_ir_time: float | None = None
    prev_ir_mean: float | None = None
    last_ball_center: np.ndarray | None = None
    last_ball_radius: float | None = None
    ball_velocity = np.zeros(2, dtype=float)

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
            if current_mean >= prev_mean:
                on_idx = frame_idx
                on_time = t
                on_gray = ir_gray
                off_gray = prev_ir_gray
            else:
                on_idx = prev_ir_idx
                on_time = prev_ir_time
                on_gray = prev_ir_gray
                off_gray = ir_gray
            if (
                on_idx not in processed_dot_frames
                and inference_start <= on_idx < inference_end
            ):
                start_pose = time.perf_counter()
                dot_detections = detect_reflective_dots(off_gray, on_gray)
                pose = dot_tracker.estimate_pose(dot_detections)
                dot_time += time.perf_counter() - start_pose
                if pose is not None:
                    rvec = pose["rvec"]
                    tvec = pose["tvec"]
                    clubface_px = pose["clubface_px"]
                    roll, pitch, yaw = rvec_to_euler(rvec)
                    record = {
                        "time": round(on_time, 3),
                        "frame": int(on_idx),
                        "clubface_center": {
                            "u": round(float(clubface_px[0]), 2),
                            "v": round(float(clubface_px[1]), 2),
                        },
                        "pose": {
                            "x": round(float(tvec[0]), 2),
                            "y": round(float(tvec[1]), 2),
                            "z": round(float(tvec[2]), 2),
                            "roll": round(float(roll), 2),
                            "pitch": round(float(pitch), 2),
                            "yaw": round(float(yaw), 2),
                        },
                        "quality": {
                            "inliers": int(pose["inliers"]),
                            "reprojection_error": round(float(pose["reprojection_error"]), 3),
                            "dots": len(dot_detections),
                        },
                    }
                    clubface_records.append(record)
                    processed_dot_frames.add(on_idx)
                    if on_idx == frame_idx:
                        cv2.circle(
                            enhanced,
                            (int(round(clubface_px[0])), int(round(clubface_px[1]))),
                            6,
                            (255, 0, 0),
                            2,
                        )
                        cv2.drawFrameAxes(
                            enhanced,
                            CAMERA_MATRIX,
                            DIST_COEFFS,
                            rvec,
                            tvec,
                            CLUBFACE_AXIS_LENGTH,
                            2,
                        )

        if frames_dir and inference_start <= frame_idx < inference_end:
            cv2.imwrite(
                os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), enhanced
            )

        prev_ir_gray = ir_gray
        prev_ir_idx = frame_idx
        prev_ir_time = t
        prev_ir_mean = current_mean
        frame_idx += 1

    cap.release()

    ball_coords.sort(key=lambda c: c["time"])
    clubface_records.sort(key=lambda c: c["time"])
    if not clubface_records:
        raise RuntimeError("No reflective dots detected in the video")

    with open(ball_path, "w", encoding="utf-8") as f:
        json.dump(ball_coords, f, indent=2)
    with open(sticker_path, "w", encoding="utf-8") as f:
        json.dump(clubface_records, f, indent=2)

    print(f"Saved {len(ball_coords)} ball points to {ball_path}")
    print(f"Saved {len(clubface_records)} clubface points to {sticker_path}")
    print(f"Ball detection compile time: {ball_compile_time:.2f}s")
    print(f"Clubface tracking compile time: {tracker_compile_time:.2f}s")
    print(f"Ball detection time: {ball_time:.2f}s")
    print(f"Clubface tracking time: {dot_time:.2f}s")
    return "skibidi"



if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "tst_good_120.mp4"
    ball_path = sys.argv[2] if len(sys.argv) > 2 else "ball_coords.json"
    sticker_path = sys.argv[3] if len(sys.argv) > 3 else "sticker_coords.json"
    frames_dir = sys.argv[4] if len(sys.argv) > 4 else "ball_frames"
    process_video(
        video_path,
        ball_path,
        sticker_path,
        frames_dir,
    )
