import json
import os
import sys
import time

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
BALL_SCORE_THRESHOLD = 0.4
MOTION_WINDOW_SCORE_THRESHOLD = 0.1
MOTION_WINDOW_MIN_ASPECT_RATIO = 0.65
MAX_CENTER_JUMP_PX = 120.0

# Load camera calibration parameters
_calib_path = os.path.join(os.path.dirname(__file__), "calibration", "camera_calib.npz")
_calib_data = np.load(_calib_path)
CAMERA_MATRIX = _calib_data["camera_matrix"]
DIST_COEFFS = _calib_data["dist_coeffs"]

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

DYNAMIC_ID = 0

MAX_MISSING_FRAMES = 12
MAX_MOTION_FRAMES = 40  # maximum allowed motion window length in frames
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
USE_BLUR = False

# Pose quality thresholds
MAX_REPROJECTION_ERROR = 2.0  # pixels
MAX_TRANSLATION_DELTA = 30.0  # translation jump threshold
MAX_ROTATION_DELTA = 45.0  # degrees


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


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    cos_theta = np.dot(q0, q1)
    if cos_theta < 0.0:
        q1 = -q1
        cos_theta = -cos_theta
    if cos_theta > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    sin_theta = np.sin(theta)
    w1 = np.sin((1.0 - alpha) * theta) / sin_theta
    w2 = np.sin(alpha * theta) / sin_theta
    return w1 * q0 + w2 * q1


def interpolate_poses(
    times: list[float],
    last_rt: tuple[np.ndarray, np.ndarray],
    curr_rt: tuple[np.ndarray, np.ndarray],
    t0: float,
    t1: float,
    out_list: list[dict],
) -> None:
    rvec0, tvec0 = last_rt
    rvec1, tvec1 = curr_rt
    # Ensure translation vectors are 1D to avoid broadcasting issues during
    # interpolation. The pose estimation functions sometimes return a column
    # vector (shape (3, 1)), while other parts of the pipeline use a flat array
    # (shape (3,)). Mixing these shapes causes numpy to broadcast the vectors
    # into a larger matrix, resulting in more than three values when unpacking.
    tvec0 = tvec0.reshape(3)
    tvec1 = tvec1.reshape(3)
    q0 = rvec_to_quat(rvec0)
    q1 = rvec_to_quat(rvec1)
    for tm in times:
        alpha = (tm - t0) / (t1 - t0)
        tvec = (1.0 - alpha) * tvec0 + alpha * tvec1
        q = slerp(q0, q1, alpha)
        rvec = quat_to_rvec(q)
        roll, pitch, yaw = rvec_to_euler(rvec)
        x, y, z = tvec.ravel()
        out_list.append(
            {
                "time": round(tm, 3),
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "z": round(float(z), 2),
                "roll": round(float(roll), 2),
                "pitch": round(float(pitch), 2),
                "yaw": round(float(yaw), 2),
            }
        )


def find_motion_window(
    video_path: str,
    detector: TFLiteBallDetector,
    *,
    pad_frames: int = MAX_MOTION_FRAMES,
    max_frames: int = MAX_MOTION_FRAMES,
    confirm_radius: int = 3,
    score_threshold: float = MOTION_WINDOW_SCORE_THRESHOLD,
) -> tuple[int, int, bool]:
    """Return a tight motion window around the ball's last appearance.

    Frames are processed sequentially so each needs at most one detector pass.
    Candidate detections must look like a ball (nearly square box, reasonable
    radius) and stay close to the previous accepted position. Once the detector
    misses ``confirm_radius`` frames in a row the search stops and the window is
    anchored around the last confirmed sighting."""

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total == 0:
        cap.release()
        return 0, 0, False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    last_detection_idx: int | None = None
    last_center: np.ndarray | None = None
    consecutive_misses = 0
    reset_after_misses = max(confirm_radius * 3, 12)

    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        enhanced, _ = preprocess_frame(frame)
        detections = detector.detect(enhanced)
        accepted = False
        if detections:
            filtered: list[tuple[float, dict]] = []
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
                _, _, radius, _ = bbox_to_ball_metrics(x1, y1, x2, y2)
                if radius < MIN_BALL_RADIUS_PX:
                    continue
                filtered.append((score, det))
            if filtered:
                _, best_det = max(filtered, key=lambda item: item[0])
                x1, y1, x2, y2 = best_det["bbox"]
                cx, cy, _, _ = bbox_to_ball_metrics(x1, y1, x2, y2)
                center = np.array([cx, cy], dtype=float)
                if last_center is not None:
                    if np.linalg.norm(center - last_center) <= MAX_CENTER_JUMP_PX:
                        accepted = True
                else:
                    accepted = True
                if accepted:
                    last_center = center
                    last_detection_idx = idx
                    consecutive_misses = 0
        if not accepted:
            consecutive_misses += 1
            if consecutive_misses > reset_after_misses:
                last_center = None
                consecutive_misses = reset_after_misses

    cap.release()

    if last_detection_idx is None:
        return 0, total, False

    start_frame = max(0, last_detection_idx - pad_frames)
    end_frame = min(total, last_detection_idx + confirm_radius + 1)
    if end_frame - start_frame > max_frames:
        start_frame = max(0, end_frame - max_frames)
    return start_frame, end_frame, True


def process_video(
    video_path: str,
    ball_path: str,
    sticker_path: str,
    frames_dir: str = "ball_frames",
) -> str:
    """Process ``video_path`` saving ball and sticker coordinates to JSON.

    Returns
    -------
    str
        The string ``"skibidi"`` when processing is complete.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ball_compile_start = time.perf_counter()
    detector = TFLiteBallDetector("golf_ball_detector.tflite", conf_threshold=0.01)
    ball_compile_time = time.perf_counter() - ball_compile_start

    start_frame, end_frame, ball_found = find_motion_window(video_path, detector)
    if not ball_found:
        raise RuntimeError("No ball detected in the video")
    print(f"Motion window frames: {start_frame}-{end_frame}")

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
    sticker_coords = []
    last_dynamic_pose = None
    last_dynamic_rt = None
    last_dynamic_quat = None
    last_valid_time = None
    tracker_corners = None
    prev_gray = None
    pending_times: list[float] = []
    missing_frames = 0
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


        sticker_start = time.perf_counter()
        corners, ids, _ = aruco_detector.detectMarkers(marker_gray)
        sticker_time += time.perf_counter() - sticker_start
        current_rt = None
        current_pose = None
        current_quat = None
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
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corner],
                        DYNAMIC_MARKER_LENGTH,
                        CAMERA_MATRIX,
                        DIST_COEFFS,
                    )
                    rvec = rvecs[0, 0]
                    # Flatten the translation vector to a 1D array for
                    # consistency with solvePnP outputs.
                    tvec = tvecs[0, 0].reshape(3)
                    x, y, z = tvec
                    curr_q = rvec_to_quat(rvec)
                    if last_dynamic_rt is not None:
                        prev_q = rvec_to_quat(last_dynamic_rt[0])
                        if np.dot(curr_q, prev_q) < 0.0:
                            curr_q = -curr_q
                            rvec = quat_to_rvec(curr_q)
                    roll, pitch, yaw = rvec_to_euler(rvec)
                    current_rt = (rvec, tvec)
                    current_pose = (x, y, z, roll, pitch, yaw)
                    current_quat = curr_q
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
            new_corners, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, marker_gray, tracker_corners, None)
            if st.sum() == 4:
                object_pts = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [DYNAMIC_MARKER_LENGTH, 0.0, 0.0],
                        [DYNAMIC_MARKER_LENGTH, DYNAMIC_MARKER_LENGTH, 0.0],
                        [0.0, DYNAMIC_MARKER_LENGTH, 0.0],
                    ],
                    dtype=np.float32,
                )
                ok, rvec, tvec = cv2.solvePnP(
                    object_pts, new_corners.reshape(-1, 2), CAMERA_MATRIX, DIST_COEFFS
                )
                if ok:
                    # ``solvePnP`` returns a column vector; flatten it so that the
                    # rest of the pipeline always works with a 1D translation
                    # vector.
                    tvec = tvec.reshape(3)
                    x, y, z = tvec
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
                        roll, pitch, yaw = rvec_to_euler(rvec)
                        current_rt = (rvec, tvec)
                        current_pose = (x, y, z, roll, pitch, yaw)
                        current_quat = curr_q
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
                        current_pose = None
                        current_quat = None
                tracker_corners = new_corners
                prev_gray = marker_gray
            else:
                tracker_corners = None
                prev_gray = None
        if current_rt is None:
            pending_times.append(t)
            if len(pending_times) > MAX_MISSING_FRAMES:
                tracker_corners = None
                prev_gray = None
                last_dynamic_pose = None
                last_dynamic_rt = None
                last_dynamic_quat = None
                last_valid_time = None
                pending_times.clear()
        else:
            if pending_times:
                if last_dynamic_rt is not None:
                    interpolate_poses(
                        pending_times,
                        last_dynamic_rt,
                        current_rt,
                        last_valid_time,
                        t,
                        sticker_coords,
                    )
                pending_times.clear()
            x, y, z, roll, pitch, yaw = current_pose
            sticker_coords.append(
                {
                    "time": round(t, 3),
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "z": round(float(z), 2),
                    "roll": round(float(roll), 2),
                    "pitch": round(float(pitch), 2),
                    "yaw": round(float(yaw), 2),
                }
            )
            last_dynamic_pose = current_pose
            last_dynamic_rt = current_rt
            last_dynamic_quat = current_quat
            last_valid_time = t
            missing_frames = 0
            if dynamic_corner is not None:
                tracker_corners = dynamic_corner.reshape(4, 1, 2).astype(np.float32)
            prev_gray = marker_gray
        if current_rt is None:
            missing_frames = len(pending_times)

        if frames_dir and inference_start <= frame_idx < inference_end:
            cv2.imwrite(
                os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), frame
            )
        frame_idx += 1

    cap.release()
    if pending_times and last_dynamic_rt is not None:
        rvec, tvec = last_dynamic_rt
        roll, pitch, yaw = rvec_to_euler(rvec)
        x, y, z = tvec.ravel()
        for tm in pending_times:
            sticker_coords.append(
                {
                    "time": round(tm, 3),
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "z": round(float(z), 2),
                    "roll": round(float(roll), 2),
                    "pitch": round(float(pitch), 2),
                    "yaw": round(float(yaw), 2),
                }
            )
        pending_times.clear()
    ball_coords.sort(key=lambda c: c["time"])
    sticker_coords.sort(key=lambda c: c["time"])
    if not sticker_coords:
        raise RuntimeError("No sticker detected in the video")
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
