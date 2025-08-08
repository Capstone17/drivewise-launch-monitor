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
from ultralytics import YOLO
import onnxruntime as ort
import torch

# Select GPU if available
DEVICE = (
    "cuda"
    if "CUDAExecutionProvider" in ort.get_available_providers()
    and torch.cuda.is_available()
    else "cpu"
)

ACTUAL_BALL_RADIUS = 2.38
FOCAL_LENGTH = 1755.0  # pixels

DYNAMIC_MARKER_LENGTH = 2.38
MIN_BALL_RADIUS_PX = 9  # pixels

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
ARUCO_PARAMS.minMarkerPerimeterRate = 0.015
ARUCO_PARAMS.maxMarkerPerimeterRate = 4.0
ARUCO_PARAMS.polygonalApproxAccuracyRate = 0.04
ARUCO_PARAMS.cornerRefinementWinSize = 9
ARUCO_PARAMS.cornerRefinementMaxIterations = 50
ARUCO_PARAMS.cornerRefinementMinAccuracy = 1e-3
ARUCO_PARAMS.adaptiveThreshConstant = 7

DYNAMIC_ID = 0

MAX_MISSING_FRAMES = 12
MAX_MOTION_FRAMES = 40  # maximum allowed motion window length in frames
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
USE_BLUR = False


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


def measure_ball(box):
    x1, y1, x2, y2 = box.xyxy[0]
    w = float(x2 - x1)
    h = float(y2 - y1)
    radius_px = (w + h) / 4.0
    cx = float(x1 + x2) / 2.0
    cy = float(y1 + y2) / 2.0
    distance = FOCAL_LENGTH * ACTUAL_BALL_RADIUS / radius_px
    return cx, cy, radius_px, distance


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
                "interpolated": True,
            }
        )


def order_corners_clockwise(pts: np.ndarray, last_pts: np.ndarray | None) -> np.ndarray:
    """Return corners in clockwise order starting from top-left.

    If ``last_pts`` is provided, choose the rotation that minimizes the distance
    to ``last_pts``.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    if last_pts is not None:
        candidates = [np.roll(ordered, k, axis=0) for k in range(4)]
        ordered = min(candidates, key=lambda p: np.linalg.norm(p - last_pts))
    return ordered


def sanitize_corners(corners: np.ndarray, last_pts: np.ndarray | None) -> np.ndarray | None:
    """Validate that ``corners`` form a plausible marker and return ordered pts."""
    pts = order_corners_clockwise(corners, last_pts)
    cross = []
    for i in range(4):
        a = pts[(i + 1) % 4] - pts[i]
        b = pts[(i + 2) % 4] - pts[(i + 1) % 4]
        cross.append(np.cross(a, b))
    cross = np.array(cross)
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return None
    sides = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
    if sides.max() / sides.min() > 1.6:
        return None
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i - 1) % 4]
        p2 = pts[(i + 1) % 4]
        v1 = p1 - p0
        v2 = p2 - p0
        ang = np.degrees(
            np.arccos(
                np.clip(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                    -1.0,
                    1.0,
                )
            )
        )
        if not 60.0 <= ang <= 120.0:
            return None
    return pts


def track_corners(
    prev_gray: np.ndarray, curr_gray: np.ndarray, last_pts: np.ndarray
) -> np.ndarray | None:
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    x0 = max(int(last_pts[:, 0].min() - 80), 0)
    y0 = max(int(last_pts[:, 1].min() - 80), 0)
    x1 = min(int(last_pts[:, 0].max() + 80), prev_gray.shape[1])
    y1 = min(int(last_pts[:, 1].max() + 80), prev_gray.shape[0])
    prev_roi = prev_gray[y0:y1, x0:x1]
    curr_roi = curr_gray[y0:y1, x0:x1]
    pts0 = (last_pts - np.array([x0, y0])).reshape(-1, 1, 2).astype(np.float32)
    pts1, st, _ = cv2.calcOpticalFlowPyrLK(prev_roi, curr_roi, pts0, None, **lk_params)
    if pts1 is not None:
        pts1 += np.array([x0, y0], dtype=np.float32)
    st = st.reshape(-1).astype(bool)
    if st.sum() < 2:
        return None
    prev_in = last_pts[st]
    curr_in = pts1.reshape(-1, 2)[st]
    H, mask = cv2.findHomography(prev_in, curr_in, cv2.RANSAC, 5.0)
    if H is not None and mask.sum() >= 2:
        pred = cv2.perspectiveTransform(last_pts.reshape(-1, 1, 2), H).reshape(4, 2)
    else:
        shift = curr_in.mean(axis=0) - prev_in.mean(axis=0)
        pred = last_pts + shift
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)
    pred = cv2.cornerSubPix(curr_gray, pred.reshape(-1, 1, 2), (7, 7), (-1, -1), criteria)
    return pred.reshape(4, 2)


def rotation_angle(rvec1: np.ndarray, rvec2: np.ndarray) -> float:
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    R = R2 @ R1.T
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)))
    return float(angle)


def solve_pose(
    img_pts: np.ndarray, last_rt: tuple[np.ndarray, np.ndarray] | None
) -> tuple[np.ndarray, np.ndarray] | None:
    object_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [DYNAMIC_MARKER_LENGTH, 0.0, 0.0],
            [DYNAMIC_MARKER_LENGTH, DYNAMIC_MARKER_LENGTH, 0.0],
            [0.0, DYNAMIC_MARKER_LENGTH, 0.0],
        ],
        dtype=np.float32,
    )
    ok, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
        object_pts,
        img_pts.reshape(-1, 2),
        CAMERA_MATRIX,
        DIST_COEFFS,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok or len(rvecs) == 0:
        return None
    errors = [np.sqrt((e ** 2).mean()) for e in reproj]
    idx = int(np.argmin(errors))
    if errors[idx] > 2.0:
        return None
    rvec = rvecs[idx]
    tvec = tvecs[idx].reshape(3)
    if last_rt is not None:
        dt = np.linalg.norm(tvec - last_rt[1].reshape(3))
        drot = rotation_angle(last_rt[0], rvec)
        if dt > 0.05 or drot > 25.0:
            return None
    return rvec, tvec

def find_motion_window(
    video_path: str,
    *,
    pad_frames: int = 60,
    max_frames: int = MAX_MOTION_FRAMES,
) -> tuple[int, int, bool]:
    """Return the frame range surrounding the dynamic sticker and whether it was found.

    The video is scanned frame-by-frame for the ArUco marker with ID
    ``DYNAMIC_ID``. The motion window spans from the first to the last frame
    where this sticker is detected, expanded by ``pad_frames`` on each side.
    The resulting range is capped to ``max_frames`` by trimming from the
    beginning. If the sticker never appears, the entire clip is returned and the
    flag is ``False``."""

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    first, last = None, None

    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = CLAHE.apply(gray)
        if USE_BLUR:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and any(m_id[0] == DYNAMIC_ID for m_id in ids):
            if first is None:
                first = idx
            last = idx

    cap.release()

    if first is None or last is None:
        return 0, total, False

    start_frame = max(0, first - pad_frames)
    end_frame = min(total, last + pad_frames)
    if end_frame - start_frame > max_frames:
        start_frame = max(end_frame - max_frames, 0)
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
    # Scan for the sticker before expensive model compilation
    start_frame, end_frame, sticker_found = find_motion_window(video_path)
    if not sticker_found:
        raise RuntimeError("No sticker detected in the video")
    print(f"Motion window frames: {start_frame}-{end_frame}")

    ball_compile_start = time.perf_counter()
    model = YOLO("golf_ball_detector.onnx", task="detect")
    ball_compile_time = time.perf_counter() - ball_compile_start

    sticker_compile_start = time.perf_counter()
    aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    sticker_compile_time = time.perf_counter() - sticker_compile_start

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    inference_start = max(0, start_frame - 5)
    inference_end = min(total_frames, end_frame + 5)
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
    last_img_pts: np.ndarray | None = None
    prev_gray = None
    pending_times: list[float] = []
    missing_frames = 0
    # Tracking of last detected ball position and velocity
    last_ball_center: np.ndarray | None = None
    last_ball_radius: float | None = None
    ball_velocity = np.zeros(2, dtype=float)
    # Tracking of last detected ball position and velocity
    last_ball_center: np.ndarray | None = None
    last_ball_radius: float | None = None
    ball_velocity = np.zeros(2, dtype=float)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig = frame
        frame, gray = preprocess_frame(frame)
        marker_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        marker_gray = CLAHE.apply(marker_gray)
        if USE_BLUR:
            marker_gray = cv2.GaussianBlur(marker_gray, (3, 3), 0)
        if h is None:
            h, w = frame.shape[:2]
        t = frame_idx / video_fps
        results = None
        in_window = start_frame <= frame_idx < end_frame
        if inference_start <= frame_idx < inference_end:
            start = time.perf_counter()
            results = model(frame, verbose=False, device=DEVICE)
            ball_time += time.perf_counter() - start

        detected = False
        detected_center: tuple[float, float, float] | None = None
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            cx, cy, rad, distance = measure_ball(boxes[best_idx])
            if rad >= MIN_BALL_RADIUS_PX:
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
                    ball_velocity = np.array([cx, cy]) - last_ball_center
                last_ball_center = np.array([cx, cy])
                last_ball_radius = rad
                detected_center = (cx, cy, rad)
                detected = True

        if (
            not detected
            and in_window
            and last_ball_center is not None
            and last_ball_radius is not None
        ):
            expected_center = last_ball_center + ball_velocity
            rate_motion = 0.1
            min_r = int(max(last_ball_radius * (1-rate_motion), MIN_BALL_RADIUS_PX - 2))
            max_r = int(last_ball_radius * (1+rate_motion))
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(5, int(last_ball_radius * 2)),
                param1=100,
                param2=15,
                minRadius=min_r,
                maxRadius=max_r,
            )
            if circles is not None:
                c = np.round(circles[0, :]).astype(int)
                ex, ey = expected_center
                # choose circle nearest expected center
                cx, cy, rad = min(
                    c,
                    key=lambda cir: (cir[0] - ex) ** 2 + (cir[1] - ey) ** 2,
                )
                dist = np.hypot(cx - ex, cy - ey)
                if dist <= 2.0 * last_ball_radius and min_r <= rad <= max_r:
                    distance = FOCAL_LENGTH * ACTUAL_BALL_RADIUS / rad
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
                    cv2.circle(frame, (int(cx), int(cy)), int(rad), (255, 0, 0), 2)
                    if last_ball_center is not None:
                        ball_velocity = np.array([cx, cy]) - last_ball_center
                    last_ball_center = np.array([cx, cy])
                    last_ball_radius = rad
                    detected_center = (cx, cy, rad)
                    detected = True

        if detected and detected_center is not None:
            cx, cy, rad = detected_center
            if (
                cx - rad <= 0
                or cy - rad <= 0
                or cx + rad >= w
                or cy + rad >= h
            ):
                print("Ball exited frame; stopping detection")
                break


                if last_ball_center is not None:
                    ball_velocity = np.array([cx, cy]) - last_ball_center
                last_ball_center = np.array([cx, cy])
                last_ball_radius = rad
                detected_center = (cx, cy, rad)
                detected = True

        if (
            not detected
            and in_window
            and last_ball_center is not None
            and last_ball_radius is not None
        ):
            expected_center = last_ball_center + ball_velocity
            rate_motion = 0.1
            min_r = int(max(last_ball_radius * (1-rate_motion), MIN_BALL_RADIUS_PX - 2))
            max_r = int(last_ball_radius * (1+rate_motion))
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(5, int(last_ball_radius * 2)),
                param1=100,
                param2=15,
                minRadius=min_r,
                maxRadius=max_r,
            )
            if circles is not None:
                c = np.round(circles[0, :]).astype(int)
                ex, ey = expected_center
                # choose circle nearest expected center
                cx, cy, rad = min(
                    c,
                    key=lambda cir: (cir[0] - ex) ** 2 + (cir[1] - ey) ** 2,
                )
                dist = np.hypot(cx - ex, cy - ey)
                if dist <= 2.0 * last_ball_radius and min_r <= rad <= max_r:
                    distance = FOCAL_LENGTH * ACTUAL_BALL_RADIUS / rad
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
                    cv2.circle(frame, (int(cx), int(cy)), int(rad), (255, 0, 0), 2)
                    if last_ball_center is not None:
                        ball_velocity = np.array([cx, cy]) - last_ball_center
                    last_ball_center = np.array([cx, cy])
                    last_ball_radius = rad
                    detected_center = (cx, cy, rad)
                    detected = True

        if detected and detected_center is not None:
            cx, cy, rad = detected_center
            if (
                cx - rad <= 0
                or cy - rad <= 0
                or cx + rad >= w
                or cy + rad >= h
            ):
                print("Ball exited frame; stopping detection")
                break


        sticker_start = time.perf_counter()
        corners, ids, _ = aruco_detector.detectMarkers(marker_gray)
        sticker_time += time.perf_counter() - sticker_start
        current_rt = None
        current_pose = None
        current_quat = None
        img_pts = None
        if ids is not None and len(ids) > 0 and DYNAMIC_ID in ids.flatten():
            idx = list(ids.flatten()).index(DYNAMIC_ID)
            corner = corners[idx].reshape(4, 2)
            cv2.aruco.drawDetectedMarkers(frame, [corners[idx]], np.array([[DYNAMIC_ID]]))
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                50,
                1e-3,
            )
            corner = cv2.cornerSubPix(
                marker_gray,
                corner.reshape(-1, 1, 2),
                (7, 7),
                (-1, -1),
                criteria,
            ).reshape(4, 2)
            img_pts = sanitize_corners(corner, last_img_pts)
        if img_pts is None and last_img_pts is not None and prev_gray is not None:
            tracked = track_corners(prev_gray, marker_gray, last_img_pts)
            if tracked is not None:
                img_pts = sanitize_corners(tracked, last_img_pts)
        if img_pts is not None:
            rt = solve_pose(img_pts, last_dynamic_rt)
            if rt is not None:
                rvec, tvec = rt
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
                cv2.drawFrameAxes(
                    frame,
                    CAMERA_MATRIX,
                    DIST_COEFFS,
                    rvec,
                    tvec,
                    DYNAMIC_MARKER_LENGTH * 0.5,
                    2,
                )
                last_img_pts = img_pts
                prev_gray = marker_gray.copy()
        if current_rt is None:
            pending_times.append(t)
            if len(pending_times) > MAX_MISSING_FRAMES:
                last_img_pts = None
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
                    "interpolated": False,
                }
            )
            last_dynamic_pose = current_pose
            last_dynamic_rt = current_rt
            last_dynamic_quat = current_quat
            last_valid_time = t
            missing_frames = 0
        if current_rt is None:
            missing_frames = len(pending_times)

        if frames_dir and inference_start <= frame_idx < inference_end:
            cv2.imwrite(
                os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), frame
            )
        frame_idx += 1

    cap.release()
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
    video_path = sys.argv[1] if len(sys.argv) > 1 else "tst_2.mp4"
    ball_path = sys.argv[2] if len(sys.argv) > 2 else "ball_coords.json"
    sticker_path = sys.argv[3] if len(sys.argv) > 3 else "sticker_coords.json"
    frames_dir = sys.argv[4] if len(sys.argv) > 4 else "ball_frames"
    process_video(
        video_path,
        ball_path,
        sticker_path,
        frames_dir,
    )
