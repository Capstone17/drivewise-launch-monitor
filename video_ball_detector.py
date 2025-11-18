from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from bisect import bisect_left

warnings.filterwarnings(
    "ignore",
    message=r"Protobuf gencode version .* is exactly one major version older .*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.lite\.Interpreter is deprecated.*",
    category=UserWarning,
    module=r"tensorflow(\.|$)",
)

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

ACTUAL_BALL_RADIUS = 2.38
FOCAL_LENGTH = 1755.0  # pixels

DYNAMIC_MARKER_LENGTH = 2.38
MIN_BALL_RADIUS_PX = 9  # pixels
EDGE_MARGIN_PX = 1
BALL_SCORE_THRESHOLD = 0.4
MOTION_WINDOW_SCORE_THRESHOLD = 0.1
MOTION_WINDOW_MIN_ASPECT_RATIO = 0.65
MAX_CENTER_JUMP_PX = 120.0
MOTION_WINDOW_FRAMES = 160  # number of frames kept in the motion window
IMPACT_SPEED_THRESHOLD_PX = 1.0  # pixel distance that marks ball movement

MOTION_WINDOW_DEBUG = os.environ.get("MOTION_WINDOW_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

try:  # NumPy < 2.0
    POLYFIT_RANK_WARNING = np.RankWarning  # type: ignore[attr-defined]
except AttributeError:  # NumPy >= 2.0
    try:
        from numpy.polynomial import polyutils as _polyutils

        POLYFIT_RANK_WARNING = _polyutils.RankWarning
    except (ImportError, AttributeError):
        class _PolyfitRankWarning(RuntimeWarning):
            """Fallback warning used when numpy.rankwarning is unavailable."""

            pass

        POLYFIT_RANK_WARNING = _PolyfitRankWarning

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

_DEFAULT_CAMERA_MATRIX = CAMERA_MATRIX.astype(np.float32, copy=True)
_DEFAULT_DIST_COEFFS = DIST_COEFFS.astype(np.float32, copy=True)
_DEFAULT_FOCAL_LENGTH = float(FOCAL_LENGTH)
_DEFAULT_BALL_RADIUS = float(ACTUAL_BALL_RADIUS)
_CURRENT_CALIBRATION: dict[str, object] | None = None


def apply_calibration(calibration: dict[str, object] | None = None) -> dict[str, object]:
    """Update global calibration parameters and return the resolved set."""

    global CAMERA_MATRIX, DIST_COEFFS, FOCAL_LENGTH, ACTUAL_BALL_RADIUS, _CURRENT_CALIBRATION

    matrix = _DEFAULT_CAMERA_MATRIX
    dist = _DEFAULT_DIST_COEFFS
    focal = _DEFAULT_FOCAL_LENGTH
    radius = _DEFAULT_BALL_RADIUS

    # if calibration:
    #     cam_mat = calibration.get("camera_matrix")
    #     if cam_mat is not None:
    #         arr = np.asarray(cam_mat, dtype=np.float32)
    #         if arr.shape == (3, 3):
    #             matrix = arr
    #     dist_vals = calibration.get("dist_coeffs")
    #     if dist_vals is not None:
    #         arr = np.asarray(dist_vals, dtype=np.float32)
    #         if arr.ndim == 1:
    #             arr = arr.reshape(1, -1)
    #         if arr.ndim == 2:
    #             dist = arr
    #     focal_val = calibration.get("focal_length")
    #     if focal_val is not None:
    #         try:
    #             focal = float(focal_val)
    #         except (TypeError, ValueError):
    #             pass
    #     radius_val = calibration.get("ball_radius")
    #     if radius_val is not None:
    #         try:
    #             radius = float(radius_val)
    #         except (TypeError, ValueError):
    #             pass

    CAMERA_MATRIX = matrix.astype(np.float32, copy=True)
    DIST_COEFFS = dist.astype(np.float32, copy=True)
    FOCAL_LENGTH = float(focal)
    ACTUAL_BALL_RADIUS = float(radius)

    _CURRENT_CALIBRATION = {
        "camera_matrix": CAMERA_MATRIX.copy(),
        "dist_coeffs": DIST_COEFFS.copy(),
        "focal_length": FOCAL_LENGTH,
        "ball_radius": ACTUAL_BALL_RADIUS,
    }
    return _CURRENT_CALIBRATION


apply_calibration()

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
ARUCO_PARAMS.adaptiveThreshWinSizeMin = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax = 53
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 4
ARUCO_PARAMS.minMarkerPerimeterRate = 0.02
ARUCO_PARAMS.polygonalApproxAccuracyRate = 0.03
ARUCO_PARAMS.cornerRefinementWinSize = 7
ARUCO_PARAMS.cornerRefinementMinAccuracy = 0.01
ARUCO_PARAMS.adaptiveThreshConstant = 7

_HAS_ARUCO_POSE_SINGLE = hasattr(cv2.aruco, "estimatePoseSingleMarkers")
_SOLVEPNP_FLAG = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
_UNIT_SQUARE_OBJECT_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

DYNAMIC_ID = 0

MAX_MISSING_FRAMES = 12
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
USE_BLUR = False

# Pose quality thresholds
MAX_REPROJECTION_ERROR = 2.0  # pixels
MAX_TRANSLATION_DELTA = 30.0  # translation jump threshold
MAX_ROTATION_DELTA = 45.0  # degrees

@dataclass
class TailCheckResult:
    ball_present: bool
    hits: int
    frames_checked: int
    scores: list[float]
    frame_indices: list[int]

def check_tail_for_ball(
    video_path: str,
    *,
    detector: TFLiteBallDetector | None = None,
    calibration: dict[str, object] | None = None,
    frames_to_check: int = 12,
    stride: int = 1,
    score_threshold: float = 0.25,
    min_hits: int = 2,
) -> TailCheckResult:
    """Inspect the tail end of a clip and report whether the ball remains visible."""

    if calibration is not None:
        apply_calibration(calibration)

    if frames_to_check <= 0:
        return TailCheckResult(False, 0, 0, [], [])
    if stride <= 0:
        raise ValueError("stride must be positive")

    owned_detector = detector is None
    if detector is None:
        detector = TFLiteBallDetector("golf_ball_detector.tflite", conf_threshold=0.01)

    frame_loop_start = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for tail check: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    scores: list[float] = []
    frame_indices: list[int] = []
    hits = 0
    processed = 0

    if total_frames > 0:
        last_idx = max(0, total_frames - 1)
        span = frames_to_check * stride
        start_idx = max(0, last_idx - span + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        current_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) or start_idx
        while processed < frames_to_check and current_idx <= last_idx:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            detections = detector.detect(frame)
            best_score = max((det.get("score", 0.0) for det in detections), default=0.0)
            scores.append(float(best_score))
            frame_indices.append(int(current_idx))
            if best_score >= score_threshold:
                hits += 1
            processed += 1
            if processed >= frames_to_check:
                break
            skip = stride - 1
            while skip > 0 and current_idx < last_idx:
                ok_skip = cap.grab()
                current_idx += 1
                if not ok_skip:
                    break
                skip -= 1
            current_idx += 1
    else:
        while processed < frames_to_check:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            detections = detector.detect(frame)
            best_score = max((det.get("score", 0.0) for det in detections), default=0.0)
            scores.append(float(best_score))
            frame_indices.append(processed)
            if best_score >= score_threshold:
                hits += 1
            processed += 1
            for _ in range(stride - 1):
                if not cap.grab():
                    break

    cap.release()
    if owned_detector and hasattr(detector, "interpreter"):
        del detector

    ball_present = hits >= min_hits and hits > 0
    return TailCheckResult(ball_present, hits, processed, scores, frame_indices)


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


def marker_object_points(marker_length: float) -> np.ndarray:
    """Return square marker object points scaled to ``marker_length``."""

    return _UNIT_SQUARE_OBJECT_POINTS * float(marker_length)


def estimate_single_marker_pose(
    marker_corners: np.ndarray,
    marker_length: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate pose of a single ArUco marker with compatibility fallbacks."""

    if marker_corners is None:
        return None
    if _HAS_ARUCO_POSE_SINGLE:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [marker_corners],
            marker_length,
            CAMERA_MATRIX,
            DIST_COEFFS,
        )
        if rvecs.size == 0 or tvecs.size == 0:
            return None
        return rvecs[0, 0], tvecs[0, 0].reshape(3)
    pts_2d = marker_corners.reshape(-1, 2).astype(np.float32)
    if pts_2d.shape[0] != 4:
        return None
    object_pts = marker_object_points(marker_length)
    ok, rvec, tvec = cv2.solvePnP(
        object_pts,
        pts_2d,
        CAMERA_MATRIX,
        DIST_COEFFS,
        flags=_SOLVEPNP_FLAG,
    )
    if not ok:
        return None
    return rvec.reshape(3), tvec.reshape(3)


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

def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a quaternion (w, x, y, z)."""
    q = np.empty(4)
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
    return q / np.linalg.norm(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rvec_to_quat(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    return rotmat_to_quat(R)


def quat_to_rvec(q: np.ndarray) -> np.ndarray:
    R = quat_to_rotmat(q)
    rvec, _ = cv2.Rodrigues(R)
    return rvec


def predict_sticker_series(
    measurements: list[dict[str, object]],
    start_frame: int,
    end_frame: int,
    fps: float,
) -> list[dict[str, float | int | str]]:

    if not measurements or start_frame >= end_frame:
        return []

    relevant = [m for m in measurements if m["frame"] < end_frame]
    relevant.sort(key=lambda m: m["frame"])
    if not relevant:
        return []

    first_frame = min(m["frame"] for m in relevant)
    series_start = max(start_frame, first_frame)
    if series_start >= end_frame:
        return []

    measured_lookup = {m["frame"]: m for m in relevant}
    measured_frames = sorted(measured_lookup)
    if not measured_frames:
        return []
    measured_positions = [
        np.array(measured_lookup[frame]["position"], dtype=float) for frame in measured_frames
    ]
    first_measured_frame = measured_frames[0]
    last_measured_frame = measured_frames[-1]
    trend_coeffs: list[np.ndarray] | None = None
    if len(measured_frames) >= 2:
        frame_values = np.array(measured_frames, dtype=float)
        position_matrix = np.vstack(measured_positions)
        trend_coeffs = []
        for axis in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", POLYFIT_RANK_WARNING)
                coeffs = np.polyfit(frame_values, position_matrix[:, axis], deg=1)
            trend_coeffs.append(coeffs)

    def extrapolate_with_trend(frame: int) -> np.ndarray:
        assert trend_coeffs is not None
        return np.array(
            [trend_coeffs[axis][0] * frame + trend_coeffs[axis][1] for axis in range(3)],
            dtype=float,
        )

    def interpolate_xyz(frame: int) -> np.ndarray:
        if len(measured_frames) == 1:
            return measured_positions[0].astype(float, copy=True)
        if frame <= first_measured_frame:
            if trend_coeffs is not None:
                return extrapolate_with_trend(frame)
            return measured_positions[0].astype(float, copy=True)
        if frame >= last_measured_frame:
            if trend_coeffs is not None:
                return extrapolate_with_trend(frame)
            return measured_positions[-1].astype(float, copy=True)
        idx = bisect_left(measured_frames, frame)
        if idx <= 0:
            left_idx = 0
            right_idx = 1
        elif idx >= len(measured_frames):
            left_idx = len(measured_frames) - 2
            right_idx = len(measured_frames) - 1
        else:
            left_idx = idx - 1
            right_idx = idx
        left_frame = measured_frames[left_idx]
        right_frame = measured_frames[right_idx]
        span = right_frame - left_frame
        if span <= 0:
            ratio = 0.0
        else:
            ratio = (frame - left_frame) / span
        left_pos = measured_positions[left_idx]
        right_pos = measured_positions[right_idx]
        return left_pos + ratio * (right_pos - left_pos)

    series = []
    for frame in range(series_start, end_frame):
        time = frame / fps
        measurement = measured_lookup.get(frame)
        if measurement is None:
            xyz = interpolate_xyz(frame)
            source = "predicted"
        else:
            pos = np.array(measurement["position"], dtype=float)
            xyz = pos
            source = "measured"
        series.append(
            {
                "frame": frame,
                "time": time,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "source": source,
            }
        )
    enforce_monotonic_predicted_z(series)
    return series


def enforce_monotonic_predicted_z(series: list[dict[str, float | int | str]]) -> None:
    """Clamp extrapolated Z segments to the last non-increasing measured value."""

    measured_indices = [idx for idx, entry in enumerate(series) if entry.get("source") == "measured"]
    if len(measured_indices) < 2:
        return

    last_non_increasing = float("inf")
    start_anchor: dict[int, float] = {}
    minima_indices: list[int] = []
    tolerance = 1e-6
    for idx in measured_indices:
        entry_z = float(series[idx]["z"])
        if entry_z <= last_non_increasing - tolerance:
            last_non_increasing = entry_z
            minima_indices.append(idx)
        else:
            last_non_increasing = min(last_non_increasing, entry_z)
        start_anchor[idx] = last_non_increasing

    for left_idx, right_idx in zip(measured_indices, measured_indices[1:]):
        if right_idx <= left_idx + 1:
            continue
        start_z = start_anchor[left_idx]
        end_z = float(series[right_idx]["z"])
        if end_z > start_z:
            end_z = start_z
        gap = right_idx - left_idx - 1
        prev_z = start_z
        for offset, idx in enumerate(range(left_idx + 1, right_idx)):
            if series[idx].get("source") == "measured":
                prev_z = float(series[idx]["z"])
                continue
            progress = (offset + 1) / (gap + 1)
            monotone_cap = start_z + (end_z - start_z) * progress
            original_z = float(series[idx]["z"])
            new_z = min(original_z, prev_z, monotone_cap)
            series[idx]["z"] = new_z
            prev_z = new_z

    last_measured_idx = measured_indices[-1]
    if last_measured_idx >= len(series) - 1:
        return

    trailing_predicted = [
        idx for idx in range(last_measured_idx + 1, len(series)) if series[idx].get("source") != "measured"
    ]
    if not trailing_predicted:
        return

    start_cap = start_anchor[last_measured_idx]
    tail_target_idx = trailing_predicted[-1]
    raw_tail_target_z = float(series[tail_target_idx]["z"])
    if raw_tail_target_z <= start_cap + tolerance:
        target_z = raw_tail_target_z
    else:
        target_z = start_cap

    start_time = float(series[last_measured_idx]["time"])
    end_time = float(series[tail_target_idx]["time"])
    duration = max(0.0, end_time - start_time)
    slope = _estimate_tail_descent_slope(series, minima_indices)
    min_drop = start_cap - target_z

    if (
        slope is None
        or duration <= 1e-6
        or min_drop <= 1e-4
        or abs(slope) <= 1e-8
    ):
        _apply_flat_tail(series, trailing_predicted, start_cap, target_z)
    else:
        _apply_shaped_tail(
            series,
            trailing_predicted,
            start_cap,
            target_z,
            start_time,
            duration,
            slope,
        )

    if raw_tail_target_z <= start_cap + tolerance:
        series[tail_target_idx]["z"] = raw_tail_target_z


def _estimate_tail_descent_slope(
    series: list[dict[str, float | int | str]],
    minima_indices: list[int],
    max_points: int = 4,
) -> float | None:
    if len(minima_indices) < 2:
        return None
    count = min(max_points, len(minima_indices))
    selected = minima_indices[-count:]
    times = np.array([float(series[idx]["time"]) for idx in selected], dtype=float)
    zs = np.array([float(series[idx]["z"]) for idx in selected], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", POLYFIT_RANK_WARNING)
        coeffs = np.polyfit(times, zs, deg=1)
    slope = float(coeffs[0])
    if slope >= -1e-6:
        return None
    return slope


def _apply_flat_tail(
    series: list[dict[str, float | int | str]],
    indices: list[int],
    start_cap: float,
    target_z: float,
) -> None:
    floor = target_z
    prev = max(start_cap, floor)
    for idx in indices[:-1]:
        original_z = float(series[idx]["z"])
        new_z = min(original_z, prev)
        new_z = max(new_z, floor)
        series[idx]["z"] = new_z
        prev = new_z
    last_idx = indices[-1]
    prev = max(prev, floor)
    series[last_idx]["z"] = floor


def _apply_shaped_tail(
    series: list[dict[str, float | int | str]],
    indices: list[int],
    start_cap: float,
    target_z: float,
    start_time: float,
    duration: float,
    slope: float,
) -> None:
    floor = target_z
    prev = max(start_cap, floor)
    for idx in indices:
        original_z = float(series[idx]["z"])
        t = float(series[idx]["time"])
        if duration > 0.0:
            u = (t - start_time) / duration
        else:
            u = 1.0
        u = max(0.0, min(1.0, u))
        hermite_z = _cubic_hermite(start_cap, floor, slope, 0.0, duration, u)
        new_z = min(original_z, prev, hermite_z)
        new_z = max(new_z, floor)
        series[idx]["z"] = new_z
        prev = new_z
    series[indices[-1]]["z"] = floor


def _cubic_hermite(
    start_val: float,
    end_val: float,
    start_slope: float,
    end_slope: float,
    duration: float,
    u: float,
) -> float:
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = (u**3 - 2.0 * u**2 + u) * duration
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = (u**3 - u**2) * duration
    return h00 * start_val + h10 * start_slope + h01 * end_val + h11 * end_slope


def annotate_interpolated_frames(
    frame_paths: dict[int, str],
    predictions: list[dict[str, float | int | str]],
) -> None:
    """Overlay interpolated sticker positions as points onto debug frames."""

    if not predictions:
        return
    for entry in predictions:
        frame_idx = entry["frame"]
        path = frame_paths.get(frame_idx)
        if not path or not os.path.exists(path):
            continue
        image = cv2.imread(path)
        if image is None:
            continue
        h, w = image.shape[:2]
        x = float(entry["x"])
        y = float(entry["y"])
        z = float(entry["z"])
        if z <= 1e-3:
            continue
        fx = CAMERA_MATRIX[0, 0]
        fy = CAMERA_MATRIX[1, 1]
        cx = CAMERA_MATRIX[0, 2]
        cy = CAMERA_MATRIX[1, 2]
        px = int(round(fx * (x / z) + cx))
        py = int(round(fy * (y / z) + cy))
        if px < 0 or py < 0 or px >= w or py >= h:
            continue
        cv2.circle(image, (px, py), 6, (0, 165, 255), -1)
        cv2.circle(image, (px, py), 9, (0, 0, 0), 2)
        cv2.imwrite(path, image)


def find_motion_window(
    video_path: str,
    detector: TFLiteBallDetector,
    *,
    pad_frames: int = MOTION_WINDOW_FRAMES,
    max_frames: int = MOTION_WINDOW_FRAMES,
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
        best_safe: dict | None = None
        best_safe_score = float("-inf")
        best_edge: dict | None = None
        best_edge_score = float("-inf")
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
            inside = True
            if frame_width and frame_height:
                inside = bbox_within_image(det["bbox"], float(frame_width), float(frame_height))
            cx, cy, radius, _ = bbox_to_ball_metrics(x1, y1, x2, y2)
            if radius < MIN_BALL_RADIUS_PX:
                continue
            center = np.array([cx, cy], dtype=float)
            candidate = {
                "frame": idx,
                "center": center,
                "radius": float(radius),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "score": float(score),
            }
            if inside:
                if score > best_safe_score:
                    best_safe = candidate
                    best_safe_score = score
            elif score > best_edge_score:
                best_edge = candidate
                best_edge_score = score
        best = best_safe if best_safe is not None else best_edge
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
    *,
    tail_check: TailCheckResult | None = None,
    calibration: dict[str, object] | None = None,
    tail_frames_to_check: int = 12,
    tail_stride: int = 1,
    tail_score_threshold: float = 0.25,
    tail_min_hits: int = 2,
) -> str:
    """Process ``video_path`` saving ball and sticker coordinates to JSON.

    Returns
    -------
    str
        The string ``"skibidi"`` when processing is complete.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    current_calibration = None
    if calibration is not None:
        current_calibration = apply_calibration(calibration)

    ball_compile_start = time.perf_counter()
    detector = TFLiteBallDetector("golf_ball_detector.tflite", conf_threshold=0.01)
    ball_compile_time = time.perf_counter() - ball_compile_start

    if tail_check is None:
        tail_check = check_tail_for_ball(
            video_path,
            detector=detector,
            calibration=current_calibration,
            frames_to_check=tail_frames_to_check,
            stride=tail_stride,
            score_threshold=tail_score_threshold,
            min_hits=tail_min_hits,
        )

    start_frame, end_frame, ball_found, motion_stats = find_motion_window(
        video_path,
        detector,
        debug=MOTION_WINDOW_DEBUG,
    )
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

    sticker_compile_start = time.perf_counter()
    aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    sticker_compile_time = time.perf_counter() - sticker_compile_start

    cap = cv2.VideoCapture(video_path)
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
    sticker_time = 0.0
    ball_coords = []
    sticker_measurements: list[dict[str, object]] = []
    saved_frame_paths: dict[int, str] = {}
    last_dynamic_rt = None
    last_dynamic_quat = None
    tracker_corners = None
    prev_gray = None
    missing_frames = 0
    impact_frame_idx: int | None = None
    impact_time: float | None = None
    # Tracking of last detected ball position and velocity
    last_ball_center: np.ndarray | None = None
    last_ball_radius: float | None = None
    ball_velocity = np.zeros(2, dtype=float)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        should_infer = inference_start <= frame_idx < inference_end
        if not should_infer:
            if frame_idx >= inference_end:
                break
            frame_idx += 1
            continue
        orig = frame
        frame, gray = preprocess_frame(frame)
        marker_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        marker_gray = CLAHE.apply(marker_gray)
        if USE_BLUR:
            marker_gray = cv2.GaussianBlur(marker_gray, (3, 3), 0)
        if h is None:
            h, w = frame.shape[:2]
        t = frame_idx / video_fps
        detections: list[dict] = []
        in_window = start_frame <= frame_idx < end_frame
        if in_window:
            start = time.perf_counter()
            detections = detector.detect(frame)
            ball_time += time.perf_counter() - start

        detected = False
        detected_center: tuple[float, float, float] | None = None
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
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.circle(frame, (int(cx), int(cy)), int(rad), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
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
                            speed = float(np.linalg.norm(ball_velocity))
                            if impact_frame_idx is None and speed >= IMPACT_SPEED_THRESHOLD_PX:
                                impact_frame_idx = frame_idx
                                impact_time = t
                        last_ball_center = center
                        last_ball_radius = rad
                        detected_center = (cx, cy, rad)
                        detected = True

        if detected and detected_center is not None:
            cx, cy, rad = detected_center
            if (
                cx <= EDGE_MARGIN_PX
                or cy <= EDGE_MARGIN_PX
                or cx >= w - EDGE_MARGIN_PX
                or cy >= h - EDGE_MARGIN_PX
            ):
                print("Ball exited frame; stopping detection")
                break


        allow_sticker = impact_frame_idx is None or frame_idx < impact_frame_idx
        if allow_sticker:
            sticker_start = time.perf_counter()
            corners, ids, _ = aruco_detector.detectMarkers(marker_gray)
            sticker_time += time.perf_counter() - sticker_start
            current_rt = None
            current_quat = None
            current_position: np.ndarray | None = None
            dynamic_corner = None
            if ids is not None and len(ids) > 0:
                valid = [
                    corners[i]
                    for i in range(len(ids))
                    if ids[i][0] == DYNAMIC_ID
                ]
                if valid:
                    valid_ids = np.array([[DYNAMIC_ID] for _ in valid])
                    cv2.aruco.drawDetectedMarkers(frame, valid, valid_ids)
                    for corner in valid:
                        pose = estimate_single_marker_pose(corner, DYNAMIC_MARKER_LENGTH)
                        if pose is None:
                            continue
                        rvec, tvec = pose
                        curr_q = rvec_to_quat(rvec)
                        if last_dynamic_rt is not None:
                            prev_q = rvec_to_quat(last_dynamic_rt[0])
                            if np.dot(curr_q, prev_q) < 0.0:
                                curr_q = -curr_q
                                rvec = quat_to_rvec(curr_q)
                        current_rt = (rvec, tvec)
                        current_quat = curr_q
                        current_position = np.array(tvec, dtype=float)
                        dynamic_corner = corner
                        cv2.drawFrameAxes(
                            frame,
                            CAMERA_MATRIX,
                            DIST_COEFFS,
                            rvec,
                            tvec,
                            DYNAMIC_MARKER_LENGTH * 0.5,
                            2,
                        )
            if current_rt is None and tracker_corners is not None and prev_gray is not None:
                new_corners, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, marker_gray, tracker_corners, None)
                if st.sum() == 4:
                    object_pts = marker_object_points(DYNAMIC_MARKER_LENGTH)
                    ok, rvec, tvec = cv2.solvePnP(
                        object_pts, new_corners.reshape(-1, 2), CAMERA_MATRIX, DIST_COEFFS
                    )
                    if ok:
                        # ``solvePnP`` returns a column vector; flatten it so that the
                        # rest of the pipeline always works with a 1D translation
                        # vector.
                        tvec = tvec.reshape(3)
                        curr_q = rvec_to_quat(rvec)
                        if last_dynamic_rt is not None:
                            prev_q = rvec_to_quat(last_dynamic_rt[0])
                            if np.dot(curr_q, prev_q) < 0.0:
                                curr_q = -curr_q
                                rvec = quat_to_rvec(curr_q)

                        # Reprojection error check
                        reproj, _ = cv2.projectPoints(
                            object_pts, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS
                        )
                        reproj_err = np.linalg.norm(
                            new_corners.reshape(-1, 2) - reproj.reshape(-1, 2), axis=1
                        ).mean()

                        pose_ok = reproj_err <= MAX_REPROJECTION_ERROR

                        # Motion delta check
                        if pose_ok and last_dynamic_rt is not None:
                            trans_delta = np.linalg.norm(tvec - last_dynamic_rt[1])
                            ang_delta = 0.0
                            if last_dynamic_quat is not None:
                                ang_delta = np.degrees(
                                    2.0
                                    * np.arccos(
                                        np.clip(np.dot(curr_q, last_dynamic_quat), -1.0, 1.0)
                                    )
                                )
                            if (
                                trans_delta > MAX_TRANSLATION_DELTA
                                or ang_delta > MAX_ROTATION_DELTA
                            ):
                                pose_ok = False

                        if pose_ok:
                            current_rt = (rvec, tvec)
                            current_quat = curr_q
                            current_position = np.array(tvec, dtype=float)
                            cv2.drawFrameAxes(
                                frame,
                                CAMERA_MATRIX,
                                DIST_COEFFS,
                                rvec,
                                tvec,
                                DYNAMIC_MARKER_LENGTH * 0.5,
                                2,
                            )
                        else:
                            current_rt = None
                            current_quat = None
                            current_position = None
                    tracker_corners = new_corners
                    prev_gray = marker_gray
                else:
                    tracker_corners = None
                    prev_gray = None
            if current_rt is None:
                missing_frames += 1
                if missing_frames > MAX_MISSING_FRAMES:
                    tracker_corners = None
                    prev_gray = None
                    last_dynamic_rt = None
                    last_dynamic_quat = None
                    missing_frames = 0
            else:
                missing_frames = 0
                sticker_measurements.append(
                    {
                        "frame": frame_idx,
                        "time": t,
                        "position": np.array(current_position, dtype=float),
                    }
                )
                last_dynamic_rt = current_rt
                last_dynamic_quat = current_quat
                if dynamic_corner is not None:
                    tracker_corners = dynamic_corner.reshape(4, 1, 2).astype(np.float32)
                prev_gray = marker_gray
        else:
            tracker_corners = None
            prev_gray = None

        if frames_dir and inference_start <= frame_idx < inference_end:
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(frame_path, frame)
            saved_frame_paths[frame_idx] = frame_path
        frame_idx += 1

    cap.release()
    if not ball_coords:
        ball_coords.append(
            {
                "time": 0.0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            }
        )
    sticker_measurements.sort(key=lambda m: m["frame"])
    if impact_frame_idx is None:
        sticker_cutoff_frame = inference_end
    else:
        sticker_cutoff_frame = min(impact_frame_idx, inference_end)
    sticker_series = predict_sticker_series(
        sticker_measurements,
        inference_start,
        sticker_cutoff_frame,
        video_fps,
    )
    sticker_coords = []
    if not sticker_series:
            sticker_coords.append(
                {
                    "time": 0,
                    "x": 0,
                    "y": 0,
                    "z": 0,
                    "label": "garbage",
                }
            )
    for entry in sticker_series:
        source = str(entry.get("source", "predicted"))
        label = "measured" if source == "measured" else "extrapolated"
        sticker_coords.append(
            {
                "time": round(entry["time"], 3),
                "x": round(float(entry["x"]), 2),
                "y": round(float(entry["y"]), 2),
                "z": round(float(entry["z"]), 2),
                "label": label,
            }
        )
    if frames_dir:
        annotate_interpolated_frames(
            saved_frame_paths,
            [entry for entry in sticker_series if entry["source"] == "predicted"],
        )
    if impact_frame_idx is not None and impact_time is not None:
        print(f"Impact frame: {impact_frame_idx} (t={impact_time:.3f}s)")
    ball_coords.sort(key=lambda c: c["time"])
    with open(ball_path, "w") as f:
        json.dump(ball_coords, f, indent=2)
    with open(sticker_path, "w") as f:
        json.dump(sticker_coords, f, indent=2)
    print(f"Saved {len(ball_coords)} ball points to {ball_path}")
    print(f"Saved {len(sticker_coords)} sticker points to {sticker_path}")
    print(f"Ball detection compile time: {ball_compile_time:.2f}s")
    print(f"Sticker detection compile time: {sticker_compile_time:.2f}s")
    print(f"Ball detection time: {ball_time:.2f}s")
    print(f"Sticker detection time: {sticker_time:.2f}s")
    return "skibidi"


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "bad_numbers_2.mp4"
    ball_path = sys.argv[2] if len(sys.argv) > 2 else "ball_coords.json"
    sticker_path = sys.argv[3] if len(sys.argv) > 3 else "sticker_coords.json"
    frames_dir = sys.argv[4] if len(sys.argv) > 4 else "ball_frames"
    process_video(
        video_path,
        ball_path,
        sticker_path,
        frames_dir,
    )
