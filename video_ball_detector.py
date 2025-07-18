import imageio.v2 as imageio
from ultralytics import YOLO
import numpy as np
import cv2
import json
import sys

ACTUAL_BALL_RADIUS = 2.135  # centimeters
FOCAL_LENGTH = 1000.0        # pixels

# Parameters for ArUco marker detection
MARKER_LENGTH = 1.75  # centimeters
CAMERA_MATRIX = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
DIST_COEFFS = np.zeros(5)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()


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


def process_video(video_path: str, ball_path: str, sticker_path: str) -> None:
    """Process ``video_path`` saving ball and sticker coordinates to JSON."""
    model = YOLO('golf_ball_detector.onnx')
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data().get('fps', 30)
    w = h = None
    ball_coords = []
    sticker_coords = []

    for idx, frame in enumerate(reader):
        if h is None:
            h, w = frame.shape[:2]
        t = idx / fps
        results = model(frame)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            cx, cy, _, distance = measure_ball(boxes[best_idx])
            bx = (cx - w / 2.0) * distance / FOCAL_LENGTH
            by = (cy - h / 2.0) * distance / FOCAL_LENGTH
            bz = distance - 30.0
            ball_coords.append({
                "time": round(t, 2),
                "x": round(bx, 2),
                "y": round(by, 2),
                "z": round(bz, 2),
            })

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, CAMERA_MATRIX, DIST_COEFFS
            )
            rvec = np.array(rvecs[0][0])
            tvec = np.array(tvecs[0][0])
            x, y, z = tvec
            roll, pitch, yaw = rvec_to_euler(rvec)
            sticker_coords.append({
                "time": round(t, 2),
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "z": round(float(z), 2),
                "roll": round(float(roll), 2),
                "pitch": round(float(pitch), 2),
                "yaw": round(float(yaw), 2),
            })

    reader.close()
    with open(ball_path, 'w') as f:
        json.dump(ball_coords, f, indent=2)
    with open(sticker_path, 'w') as f:
        json.dump(sticker_coords, f, indent=2)
    print(f'Saved {len(ball_coords)} ball points to {ball_path}')
    print(f'Saved {len(sticker_coords)} sticker points to {sticker_path}')


if __name__ == '__main__':
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'video.mp4'
    ball_path = sys.argv[2] if len(sys.argv) > 2 else 'ball_coords.json'
    sticker_path = sys.argv[3] if len(sys.argv) > 3 else 'sticker_coords.json'
    process_video(video_path, ball_path, sticker_path)
