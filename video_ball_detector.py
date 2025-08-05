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
STATIONARY_MARKER_LENGTH = 3.5
MIN_BALL_RADIUS_PX = 9  # pixels

# Load camera calibration parameters
_calib_path = os.path.join(os.path.dirname(__file__), "calibration", "camera_calib.npz")
_calib_data = np.load(_calib_path)
CAMERA_MATRIX = _calib_data["K"]
DIST_COEFFS = _calib_data["dist"]

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

STATIONARY_ID = 1
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


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = CLAHE.apply(gray)
    if USE_BLUR:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


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
    *,
    pad_frames: int = 20,
    max_frames: int = MAX_MOTION_FRAMES,
) -> tuple[int, int]:
    """Return the frame range surrounding the dynamic sticker.

    The video is scanned frame-by-frame for the ArUco marker with ID
    ``DYNAMIC_ID``. The motion window spans from the first to the last frame
    where this sticker is detected, expanded by ``pad_frames`` on each side.
    The resulting range is capped to ``max_frames`` by trimming from the
    beginning. If the sticker never appears, the entire clip is returned."""

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    first, last = None, None

    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        gray = preprocess_frame(frame)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and any(m_id[0] == DYNAMIC_ID for m_id in ids):
            if first is None:
                first = idx
            last = idx

    cap.release()

    if first is None or last is None:
        return 0, total

    start_frame = max(0, first - pad_frames)
    end_frame = min(total, last + pad_frames)
    if end_frame - start_frame > max_frames:
        start_frame = max(end_frame - max_frames, 0)
    return start_frame, end_frame


def process_video(
    video_path: str,
    ball_path: str,
    sticker_path: str,
    stationary_path: str,
    output_path: str = "annotated_output.mp4",
    frames_dir: str = "ball_frames",
) -> None:
    """Process ``video_path`` saving ball and sticker coordinates to JSON.

    ``stationary_path`` stores the averaged pose of the stationary marker."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    ball_compile_start = time.perf_counter()
    model = YOLO("golf_ball_detector.onnx", task="detect")
    ball_compile_time = time.perf_counter() - ball_compile_start

    sticker_compile_start = time.perf_counter()
    aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    sticker_compile_time = time.perf_counter() - sticker_compile_start

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer_fps = video_fps
    # mpeg4 codecs fail when the timebase denominator exceeds 65535
    if writer_fps * 1000 > 65535:
        writer_fps = 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    start_frame, end_frame = 0, total_frames
    if total_frames > 0:
        start_frame, end_frame = find_motion_window(video_path)
        print(f"Motion window frames: {start_frame}-{end_frame}")
    inference_start = max(0, start_frame - 5)
    inference_end = min(total_frames, end_frame + 5)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    writer = None
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
    stationary_sum = np.zeros(6, dtype=float)
    stationary_count = 0
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
        if h is None:
            h, w = frame.shape[:2]
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, writer_fps, (w, h))
        t = frame_idx / video_fps
        gray = preprocess_frame(frame)
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


        sticker_start = time.perf_counter()
        corners, ids, _ = aruco_detector.detectMarkers(gray)
        sticker_time += time.perf_counter() - sticker_start
        current_rt = None
        current_pose = None
        current_quat = None
        dynamic_corner = None
        if ids is not None and len(ids) > 0:
            valid = [
                (corners[i], ids[i][0])
                for i in range(len(ids))
                if ids[i][0] in (STATIONARY_ID, DYNAMIC_ID)
            ]
            if valid:
                valid_corners = [c for c, _ in valid]
                valid_ids = np.array([[mid] for _, mid in valid])
                cv2.aruco.drawDetectedMarkers(frame, valid_corners, valid_ids)
                for corner, marker_id in valid:
                    length = (
                        STATIONARY_MARKER_LENGTH
                        if marker_id == STATIONARY_ID
                        else DYNAMIC_MARKER_LENGTH
                    )
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corner],
                        length,
                        CAMERA_MATRIX,
                        DIST_COEFFS,
                    )
                    rvec = rvecs[0, 0]
                    tvec = tvecs[0, 0]
                    x, y, z = tvec
                    if marker_id == STATIONARY_ID:
                        roll, pitch, yaw = rvec_to_euler(rvec)
                        stationary_sum += (x, y, z, roll, pitch, yaw)
                        stationary_count += 1
                    else:
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
                        length * 0.5,
                        2,
                    )
        if current_rt is None and tracker_corners is not None and prev_gray is not None:
            new_corners, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, tracker_corners, None)
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
                    x, y, z = tvec.ravel()
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
                tracker_corners = new_corners
                prev_gray = gray
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
            prev_gray = gray
        if current_rt is None:
            missing_frames = len(pending_times)

        if frames_dir and inference_start <= frame_idx < inference_end:
            cv2.imwrite(
                os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), frame
            )
        if writer is not None:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    ball_coords.sort(key=lambda c: c["time"])
    sticker_coords.sort(key=lambda c: c["time"])
    if not sticker_coords:
        raise RuntimeError("No sticker detected in the video")
    with open(ball_path, "w") as f:
        json.dump(ball_coords, f, indent=2)
    with open(sticker_path, "w") as f:
        json.dump(sticker_coords, f, indent=2)
    stationary_output = []
    if stationary_count:
        avg = stationary_sum / stationary_count
        stationary_output.append(
            {
                "x": round(float(avg[0]), 2),
                "y": round(float(avg[1]), 2),
                "z": round(float(avg[2]), 2),
                "roll": round(float(avg[3]), 2),
                "pitch": round(float(avg[4]), 2),
                "yaw": round(float(avg[5]), 2),
            }
        )
    with open(stationary_path, "w") as f:
        json.dump(stationary_output, f, indent=2)
    print(f"Saved {len(ball_coords)} ball points to {ball_path}")
    print(f"Saved {len(sticker_coords)} sticker points to {sticker_path}")
    if stationary_count:
        print(f"Saved averaged stationary sticker pose to {stationary_path}")
    print(f"Ball detection compile time: {ball_compile_time:.2f}s")
    print(f"Sticker detection compile time: {sticker_compile_time:.2f}s")
    print(f"Ball detection time: {ball_time:.2f}s")
    print(f"Sticker detection time: {sticker_time:.2f}s")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "exposure_test/tst_skinny_240.mp4"
    ball_path = sys.argv[2] if len(sys.argv) > 2 else "ball_coords.json"
    sticker_path = sys.argv[3] if len(sys.argv) > 3 else "sticker_coords.json"
    stationary_path = sys.argv[4] if len(sys.argv) > 4 else "stationary_sticker.json"
    output_path = sys.argv[5] if len(sys.argv) > 5 else "annotated_output.mp4"
    frames_dir = sys.argv[6] if len(sys.argv) > 6 else "ball_frames"
    process_video(
        video_path,
        ball_path,
        sticker_path,
        stationary_path,
        output_path,
        frames_dir,
    )
