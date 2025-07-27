import json
import sys
import time

import cv2
import numpy as np
try:
    import cv2.dnn as cvdnn
    from cv2.dnn import Net as cvdnn_Net
except Exception:  # pragma: no cover - older opencv
    cvdnn = cv2.dnn
    cvdnn_Net = cv2.dnn_Net

ACTUAL_BALL_RADIUS = 21.35  # milimeters
FOCAL_LENGTH = 1900.0  # pixels

DYNAMIC_MARKER_LENGTH = 1.75  # milimeters (club sticker)
STATIONARY_MARKER_LENGTH = 65  # milimeters (block sticker)
CAMERA_MATRIX = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
DIST_COEFFS = np.zeros(5)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# ID of the stationary ArUco marker
STATIONARY_ID = 1

# Parameters for ONNX ball detector
IMG_SIZE = 512
STRIDES = [8, 16, 32]


def _make_grid(img_size=IMG_SIZE, strides=STRIDES):
    """Create grid and stride arrays for YOLOv8 decoding."""
    grid_list = []
    stride_list = []
    for s in strides:
        num = img_size // s
        y, x = np.meshgrid(np.arange(num), np.arange(num))
        grid_list.append(np.stack((x, y), axis=-1).reshape(-1, 2))
        stride_list.append(np.full((num * num, 1), s, dtype=float))
    grid = np.concatenate(grid_list, axis=0).astype(float)
    stride = np.concatenate(stride_list, axis=0).astype(float)
    return grid, stride


GRID, STRIDE = _make_grid()


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


def detect_ball(net: cvdnn_Net, frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Run ONNX model and return bounding boxes in xyxy format with confidence."""
    blob = cvdnn.blobFromImage(frame, 1.0 / 255.0, (IMG_SIZE, IMG_SIZE), swapRB=True)
    net.setInput(blob)
    pred = net.forward().reshape(5, -1).T

    xy = (pred[:, :2] * 2 + GRID) * STRIDE
    wh = (pred[:, 2:4] * 2) ** 2 * STRIDE
    conf = 1 / (1 + np.exp(-pred[:, 4:5]))
    xyxy = np.concatenate([xy - wh / 2, xy + wh / 2, conf], axis=1)
    scale = np.array([[w / IMG_SIZE, h / IMG_SIZE, w / IMG_SIZE, h / IMG_SIZE, 1]])
    return xyxy * scale


def process_video(
    video_path: str,
    ball_path: str,
    sticker_path: str,
    stationary_path: str,
) -> None:
    """Process ``video_path`` saving ball and sticker coordinates to JSON.

    ``stationary_path`` stores the averaged pose of the stationary marker."""
    net = cvdnn.readNetFromONNX("golf_ball_detector.onnx")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    print("fps: ", fps, "width: ", w, "height :", h)
    ball_coords = []
    sticker_coords = []
    stationary_sum = np.zeros(6, dtype=float)
    stationary_count = 0
    ball_time = 0.0
    sticker_time = 0.0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if h is None:
            h, w = frame.shape[:2]
        t = frame_idx / fps
        frame_idx += 1
        start = time.perf_counter()
        boxes = detect_ball(net, frame, w, h)
        ball_time += time.perf_counter() - start
        if boxes.size > 0:
            best_idx = boxes[:, 4].argmax()
            cx, cy, _, distance = measure_ball(tuple(boxes[best_idx, :4]))
            bx = (cx - w / 2.0) * distance / FOCAL_LENGTH
            by = (cy - h / 2.0) * distance / FOCAL_LENGTH
            bz = distance - 30.0
            ball_coords.append(
                {
                    "time": round(t, 2),
                    "x": round(bx, 2),
                    "y": round(by, 2),
                    "z": round(bz, 2),
                }
            )

        sticker_start = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                length = (
                    STATIONARY_MARKER_LENGTH
                    if marker_id == STATIONARY_ID
                    else DYNAMIC_MARKER_LENGTH
                )
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[i]],
                    length,
                    CAMERA_MATRIX,
                    DIST_COEFFS,
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
        sticker_time += time.perf_counter() - sticker_start

    cap.release()
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
    print(f"Ball detection time: {ball_time:.2f}s")
    print(f"Sticker detection time: {sticker_time:.2f}s")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "video_ball_and_sticker.mp4"
    ball_path = sys.argv[2] if len(sys.argv) > 2 else "ball_coords.json"
    sticker_path = sys.argv[3] if len(sys.argv) > 3 else "sticker_coords.json"
    stationary_path = sys.argv[4] if len(sys.argv) > 4 else "stationary_sticker.json"
    process_video(video_path, ball_path, sticker_path, stationary_path)
