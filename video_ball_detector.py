import json
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

ACTUAL_BALL_RADIUS = 21.35  # milimeters
FOCAL_LENGTH = 1900.0  # pixels

DYNAMIC_MARKER_LENGTH = 1.75  # milimeters (club sticker)
STATIONARY_MARKER_LENGTH = 65  # milimeters (block sticker)
CAMERA_MATRIX = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
DIST_COEFFS = np.zeros(5)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# ID of the stationary ArUco marker
STATIONARY_ID = 1



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


def measure_ball(box: tuple[float, float, float, float]):
    """Return center and distance from bounding box coordinates."""
    x1, y1, x2, y2 = box
    w = float(x2 - x1)
    h = float(y2 - y1)
    radius_px = (w + h) / 4.0
    cx = float(x1 + x2) / 2.0
    cy = float(y1 + y2) / 2.0
    distance = FOCAL_LENGTH * ACTUAL_BALL_RADIUS / radius_px
    return cx, cy, radius_px, distance


def estimate_pose_single_markers(corners: list[np.ndarray], length: float):
    """Fallback for ``cv2.aruco.estimatePoseSingleMarkers``."""
    obj = np.array(
        [
            [-length / 2, length / 2, 0],
            [length / 2, length / 2, 0],
            [length / 2, -length / 2, 0],
            [-length / 2, -length / 2, 0],
        ],
        dtype=np.float32,
    )
    rvecs = []
    tvecs = []
    for c in corners:
        ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2), CAMERA_MATRIX, DIST_COEFFS)
        if not ok:
            rvec = np.zeros(3)
            tvec = np.zeros(3)
        rvecs.append(rvec.reshape(1, 3))
        tvecs.append(tvec.reshape(1, 3))
    return np.array(rvecs), np.array(tvecs), None



def remove_outliers(points: list[dict]) -> list[dict]:
    """Remove outliers from pixel coordinates using MAD."""
    if not points:
        return points
    arr = np.array([[p["cx"], p["cy"]] for p in points], dtype=float)
    med = np.median(arr, axis=0)
    diff = np.linalg.norm(arr - med, axis=1)
    mad = np.median(diff)
    if mad == 0:
        return points
    mask = diff < 2.5 * mad
    return [p for p, m in zip(points, mask) if m]


def fit_pixel_curve(points: list[dict]):
    """Fit linear curves for pixel centers over time."""
    t = np.array([p["time"] for p in points], dtype=float)
    cx = np.array([p["cx"] for p in points], dtype=float)
    cy = np.array([p["cy"] for p in points], dtype=float)
    px = np.polyfit(t, cx, 1)
    py = np.polyfit(t, cy, 1)
    return px, py


def detect_with_hough(
    video_path: str,
    coeffs: tuple[np.ndarray, np.ndarray],
    fps: float,
    w: int,
    h: int,
    search_radius: int,
) -> list[dict]:
    """Detect circles along the approximated curve within ``search_radius``."""
    px, py = coeffs
    cap = cv2.VideoCapture(video_path)
    coords: list[dict] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        frame_idx += 1
        cx_pred = int(np.polyval(px, t))
        cy_pred = int(np.polyval(py, t))
        x1 = max(0, cx_pred - search_radius)
        y1 = max(0, cy_pred - search_radius)
        x2 = min(w, cx_pred + search_radius)
        y2 = min(h, cy_pred + search_radius)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=search_radius // 2,
            param1=100,
            param2=25,
            minRadius=5,
            maxRadius=50,
        )
        if circles is None:
            continue
        circles = np.round(circles[0, :]).astype(int)
        cx_roi, cy_roi, r = circles[0]
        cx = cx_roi + x1
        cy = cy_roi + y1
        distance = FOCAL_LENGTH * ACTUAL_BALL_RADIUS / r
        bx = (cx - w / 2.0) * distance / FOCAL_LENGTH
        by = (cy - h / 2.0) * distance / FOCAL_LENGTH
        bz = distance - 30.0
        coords.append(
            {
                "time": round(t, 2),
                "x": round(bx, 2),
                "y": round(by, 2),
                "z": round(bz, 2),
            }
        )
    cap.release()
    return coords


def process_video(
    video_path: str,
    ball_path: str,
    sticker_path: str,
    stationary_path: str,
    annotated_path: str = "annotated.mp4",
    yolo_interval: int = 10,
    search_radius: int = 40,
    curve_path: str = "curve_path.json",
) -> None:
    """Process ``video_path`` saving ball, sticker and path coordinates to JSON.

    ``stationary_path`` stores the averaged pose of the stationary marker.
    Prints compile and runtime statistics for each detector and saves an
    annotated video. YOLOv8 inference is performed every ``yolo_interval``
    frames (default 10). After initial detection an approximated trajectory is
    fitted and OpenCV circle detection is performed within ``search_radius``
    pixels of that curve. The approximated curve is saved to ``curve_path``."""
    ball_compile_start = time.perf_counter()
    model = YOLO("golf_ball_detector.onnx")
    ball_compile_time = time.perf_counter() - ball_compile_start

    sticker_compile_start = time.perf_counter()
    aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    sticker_compile_time = time.perf_counter() - sticker_compile_start
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    print("fps: ", fps, "width: ", w, "height :", h)
    annotated_frames: list[np.ndarray] = []
    sticker_coords = []
    stationary_sum = np.zeros(6, dtype=float)
    stationary_count = 0
    ball_time = 0.0
    sticker_time = 0.0
    yolo_frames = 0
    circle_frames = 0
    ball_coords: list[dict] = []
    yolo_points: list[dict] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if h is None:
            h, w = frame.shape[:2]
        t = frame_idx / fps
        frame_idx += 1

        if frame_idx % yolo_interval == 0:
            start = time.perf_counter()
            results = model(frame, verbose=False, imgsz=512)
            ball_time += time.perf_counter() - start
            yolo_frames += 1
        else:
            results = None
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            box_xyxy = tuple(boxes[best_idx].xyxy[0])
            cx, cy, r, distance = measure_ball(box_xyxy)
            yolo_points.append({"time": t, "cx": cx, "cy": cy})
            x1, y1, x2, y2 = map(int, box_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sticker_start = time.perf_counter()
        corners, ids, _ = aruco_detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                length = (
                    STATIONARY_MARKER_LENGTH
                    if marker_id == STATIONARY_ID
                    else DYNAMIC_MARKER_LENGTH
                )
                rvecs, tvecs, _ = estimate_pose_single_markers(
                    [corners[i]],
                    length,
                )
                rvec = rvecs[0, 0]
                tvec = tvecs[0, 0]
                x, y, z = tvec
                roll, pitch, yaw = rvec_to_euler(rvec)
                if marker_id == STATIONARY_ID:
                    stationary_sum += (x, y, z, roll, pitch, yaw)
                    stationary_count += 1
                else:
                    sticker_coords.append(
                        {
                            "time": round(t, 2),
                            "x": round(float(x), 2),
                            "y": round(float(y), 2),
                            "z": round(float(z), 2),
                            "roll": round(float(roll), 2),
                            "pitch": round(float(pitch), 2),
                            "yaw": round(float(yaw), 2),
                        }
                    )
                # Axis drawing requires aruco contrib module which may not be available
        sticker_time += time.perf_counter() - sticker_start
        annotated_frames.append(frame.copy())

    cap.release()

    # Refine the YOLO detections
    filtered = remove_outliers(yolo_points)
    coeffs = fit_pixel_curve(filtered) if filtered else (np.array([0, 0]), np.array([0, 0]))
    ball_coords = detect_with_hough(
        video_path,
        coeffs,
        fps,
        w,
        h,
        search_radius,
    )
    circle_frames = len(ball_coords)

    # Draw the approximated trajectory on the annotated frames
    px, py = coeffs
    writer = cv2.VideoWriter(
        annotated_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    curve_points: list[tuple[int, int]] = []
    path_points: list[dict] = []
    for idx in range(len(annotated_frames)):
        t = idx / fps
        cx_pred_float = float(np.polyval(px, t))
        cy_pred_float = float(np.polyval(py, t))
        curve_points.append((int(cx_pred_float), int(cy_pred_float)))
        path_points.append(
            {
                "time": round(t, 2),
                "cx": round(cx_pred_float, 2),
                "cy": round(cy_pred_float, 2),
            }
        )

    for idx, frame in enumerate(annotated_frames):
        pts = np.array(curve_points[: idx + 1], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(frame, [pts], False, (0, 0, 255), 2)
        else:
            cv2.circle(frame, curve_points[0], 3, (0, 0, 255), -1)
        writer.write(frame)
    writer.release()

    with open(curve_path, "w") as f:
        json.dump(path_points, f, indent=2)

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
    print(f"YOLO frames: {yolo_frames}")
    print(f"Circle frames: {circle_frames}")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "video_ball_and_sticker.mp4"
    ball_path = sys.argv[2] if len(sys.argv) > 2 else "ball_coords.json"
    sticker_path = sys.argv[3] if len(sys.argv) > 3 else "sticker_coords.json"
    stationary_path = sys.argv[4] if len(sys.argv) > 4 else "stationary_sticker.json"
    annotated_path = sys.argv[5] if len(sys.argv) > 5 else "annotated.mp4"
    yolo_interval = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    search_radius = int(sys.argv[7]) if len(sys.argv) > 7 else 40
    curve_path = sys.argv[8] if len(sys.argv) > 8 else "curve_path.json"
    process_video(
        video_path,
        ball_path,
        sticker_path,
        stationary_path,
        annotated_path,
        yolo_interval,
        search_radius,
        curve_path,
    )
