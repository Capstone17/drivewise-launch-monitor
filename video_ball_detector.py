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
DEVICE = "cpu"  # Force CPU inference to avoid device mismatch errors

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

# Visualization and debug I/O (disable for speed)
VISUALIZE = False

# Runtime tuning
REANCHOR_INTERVAL = 12  # frames between full ArUco detections when tracking OK
BALL_REACQUIRE_INTERVAL = 10  # frames between full-frame YOLO checks when ROI-tracking

# Encourage efficient threading in ORT/OpenMP on CPU
_threads = str(min(8, max(1, (os.cpu_count() or 4))))
os.environ.setdefault("OMP_NUM_THREADS", _threads)
os.environ.setdefault("MKL_NUM_THREADS", _threads)
os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("ORT_NUM_THREADS", _threads)

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
            }
        )


def find_motion_window(
    video_path: str,
    *,
    pad_frames: int = 20,
    max_frames: int = MAX_MOTION_FRAMES,
) -> tuple[int, int, bool]:
    """Return the frame range surrounding the dynamic sticker and whether it was found.

    Faster, equally reliable approach using a coarse-to-fine scan:
    - Coarse passes sample frames with a stride (using ``grab`` to skip decoding)
      to quickly approximate the first/last detection.
    - Boundary refinement scans only a small neighborhood with step 1 to find the
      exact first/last frames.
    This reduces ArUco calls from O(N) to about O(N/stride) while preserving
    accuracy.
    """

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return 0, 0, False

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

    def detect_has_dynamic(gray_img: np.ndarray) -> bool:
        corners, ids, _ = detector.detectMarkers(gray_img)
        return ids is not None and any(m_id[0] == DYNAMIC_ID for m_id in ids)

    # Choose a conservative stride to retain reliability.
    # At 120 FPS, stride=8 samples ~15 times per second.
    stride = 8

    # Helper: get grayscale (CLAHE + optional blur)
    def to_gray(img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = CLAHE.apply(g)
        if USE_BLUR:
            g = cv2.GaussianBlur(g, (3, 3), 0)
        return g

    # First coarse pass: offset 0
    coarse_first, coarse_last = None, None
    idx = 0
    while idx < total:
        ret, frame = cap.read()
        if not ret:
            break
        gray = to_gray(frame)
        if detect_has_dynamic(gray):
            if coarse_first is None:
                coarse_first = idx
            coarse_last = idx
        # Skip next stride-1 frames cheaply
        skipped = 0
        while skipped < stride - 1 and (idx + 1) < total:
            if not cap.grab():
                break
            skipped += 1
            idx += 1
        idx += 1

    # Second coarse pass with half-stride offset to reduce aliasing risk
    offset = stride // 2
    if offset > 0 and total > offset:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Fast-forward to offset using grab
        for _ in range(offset):
            if not cap.grab():
                break
        idx = offset
        while idx < total:
            ret, frame = cap.read()
            if not ret:
                break
            gray = to_gray(frame)
            if detect_has_dynamic(gray):
                if coarse_first is None or idx < coarse_first:
                    coarse_first = idx
                if coarse_last is None or idx > coarse_last:
                    coarse_last = idx
            skipped = 0
            while skipped < stride - 1 and (idx + 1) < total:
                if not cap.grab():
                    break
                skipped += 1
                idx += 1
            idx += 1

    # If still not found, give up early
    if coarse_first is None or coarse_last is None:
        cap.release()
        return 0, total, False

    # Refine start boundary (scan a small neighborhood with step 1)
    refine_start = max(0, coarse_first - (stride - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, refine_start)
    refined_first = None
    for idx in range(refine_start, min(coarse_first + 1, total)):
        ret, frame = cap.read()
        if not ret:
            break
        if detect_has_dynamic(to_gray(frame)):
            refined_first = idx
            break
    if refined_first is None:
        refined_first = coarse_first

    # Refine end boundary. Start at coarse_last and walk forward a limited range
    # to catch detections between coarse samples. Stop if we see a gap larger
    # than stride without detections.
    refined_last = coarse_last
    cap.set(cv2.CAP_PROP_POS_FRAMES, coarse_last)
    consecutive_misses = 0
    max_forward_scan = min(total - coarse_last, stride * 2)
    for step in range(max_forward_scan):
        ret, frame = cap.read()
        if not ret:
            break
        if detect_has_dynamic(to_gray(frame)):
            refined_last = coarse_last + step
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            if consecutive_misses >= stride:
                break

    cap.release()

    start_frame = max(0, refined_first - pad_frames)
    end_frame = min(total, refined_last + pad_frames)
    if end_frame - start_frame > max_frames:
        start_frame = max(end_frame - max_frames, 0)
    return start_frame, end_frame, True


def process_video(
    video_path: str,
    ball_path: str,
    sticker_path: str,
    frames_dir: str | None = "ball_frames",
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
        print(f"Saving annotated frames to {frames_dir} from {inference_start} to {inference_end - 1}")
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
    ball_roi: tuple[int, int, int, int] | None = None  # x1,y1,x2,y2 in full-frame coords
    last_ball_detect_idx = -10**9

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
        # Draw overlays if visualization is enabled or if we're saving frames
        should_draw = bool(VISUALIZE or (frames_dir and (inference_start <= frame_idx < inference_end)))
        results = None
        in_window = start_frame <= frame_idx < end_frame
        if in_window:
            start = time.perf_counter()
            do_full = (
                ball_roi is None
                or (frame_idx - last_ball_detect_idx) >= BALL_REACQUIRE_INTERVAL
            )
            roi_offset = (0, 0)
            if not do_full and last_ball_center is not None and last_ball_radius is not None:
                # Predict next center using simple constant-velocity model
                pred = last_ball_center + ball_velocity
                margin = int(np.clip(last_ball_radius * 4.0, 64, max(64, min(w, h) // 2)))
                x1 = max(0, int(pred[0] - margin))
                y1 = max(0, int(pred[1] - margin))
                x2 = min(w, int(pred[0] + margin))
                y2 = min(h, int(pred[1] + margin))
                if x2 - x1 > 10 and y2 - y1 > 10:
                    crop = frame[y1:y2, x1:x2]
                    results = model(crop, verbose=False, device="cpu")
                    roi_offset = (x1, y1)
            # Fallback to full frame when needed
            if results is None or len(results[0].boxes) == 0:
                results = model(frame, verbose=False, device="cpu")
                roi_offset = (0, 0)
                do_full = True
            ball_time += time.perf_counter() - start

        detected = False
        detected_center: tuple[float, float, float] | None = None
        if in_window and results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            cx, cy, rad, distance = measure_ball(boxes[best_idx])
            # Adjust for ROI offset when using cropped inference
            cx += roi_offset[0]
            cy += roi_offset[1]
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
                if should_draw:
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
                last_ball_detect_idx = frame_idx
                # Update ROI around new center
                pred = last_ball_center + ball_velocity
                margin = int(np.clip(last_ball_radius * 4.0, 64, max(64, min(w, h) // 2)))
                x1 = max(0, int(pred[0] - margin))
                y1 = max(0, int(pred[1] - margin))
                x2 = min(w, int(pred[0] + margin))
                y2 = min(h, int(pred[1] + margin))
                if x2 - x1 > 10 and y2 - y1 > 10:
                    ball_roi = (x1, y1, x2, y2)

        if detected and detected_center is not None:
            cx, cy, rad = detected_center
            if (
                cx - rad <= 0
                or cy - rad <= 0
                or cx + rad >= w
                or cy + rad >= h
            ):
                # Save the current frame before exiting, if requested
                if frames_dir and inference_start <= frame_idx < inference_end:
                    cv2.imwrite(
                        os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), frame
                    )
                print("Ball exited frame; stopping detection")
                break


        if in_window:
            sticker_step_start = time.perf_counter()
            current_rt = None
            current_pose = None
            current_quat = None
            dynamic_corner = None

            # Prefer fast KLT tracking + PnP when we already have corners, and only
            # re-run full ArUco detection periodically or when KLT is unreliable.
            use_klt = (
                tracker_corners is not None
                and prev_gray is not None
                and (last_dynamic_rt is not None)
                and (frame_idx - (int(last_valid_time * video_fps) if last_valid_time is not None else 0) < REANCHOR_INTERVAL)
            )

            if use_klt:
                new_corners, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, marker_gray, tracker_corners, None
                )
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
                            dynamic_corner = new_corners
                            if should_draw:
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
                            # Force re-detection this frame
                            tracker_corners = None
                            prev_gray = None
                    else:
                        tracker_corners = None
                        prev_gray = None
                else:
                    tracker_corners = None
                    prev_gray = None

            if current_rt is None:
                # Full ArUco detection (slower) to (re)anchor tracking
                corners, ids, _ = aruco_detector.detectMarkers(marker_gray)
                if ids is not None and len(ids) > 0:
                    valid = [
                        corners[i]
                        for i in range(len(ids))
                        if ids[i][0] == DYNAMIC_ID
                    ]
                    if valid:
                        if should_draw:
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
                            dynamic_corner = corner.reshape(4, 1, 2).astype(np.float32)
                            if should_draw:
                                cv2.drawFrameAxes(
                                    frame,
                                    CAMERA_MATRIX,
                                    DIST_COEFFS,
                                    rvec,
                                    tvec,
                                    DYNAMIC_MARKER_LENGTH * 0.5,
                                    2,
                                )
                            break

            sticker_time += time.perf_counter() - sticker_step_start
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
                    # Ensure shape (4,1,2) for LK
                    if dynamic_corner.shape == (4, 2):
                        tracker_corners = dynamic_corner.reshape(4, 1, 2).astype(np.float32)
                    else:
                        tracker_corners = dynamic_corner.astype(np.float32)
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
    video_path = sys.argv[1] if len(sys.argv) > 1 else "440_model8_36s_170cm.mp4"
    ball_path = sys.argv[2] if len(sys.argv) > 2 else "ball_coords.json"
    sticker_path = sys.argv[3] if len(sys.argv) > 3 else "sticker_coords.json"
    frames_dir = sys.argv[4] if len(sys.argv) > 4 else "ball_frames"
    process_video(
        video_path,
        ball_path,
        sticker_path,
        frames_dir,
    )
