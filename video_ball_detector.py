import json
import sys

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
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

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


def measure_ball(box):
    x1, y1, x2, y2 = box.xyxy[0]
    w = float(x2 - x1)
    h = float(y2 - y1)
    radius_px = (w + h) / 4.0
    cx = float(x1 + x2) / 2.0
    cy = float(y1 + y2) / 2.0
    distance = FOCAL_LENGTH * ACTUAL_BALL_RADIUS / radius_px
    return cx, cy, radius_px, distance


def process_video(
    video_path: str,
    ball_path: str,
    sticker_path: str,
    stationary_path: str,
    output_path: str = "annotated_output.mp4",
) -> None:
    """Process ``video_path`` saving ball and sticker coordinates to JSON.

    ``stationary_path`` stores the averaged pose of the stationary marker.
    ``output_path`` is the video with all detections drawn."""
    model = YOLO("golf_ball_detector.onnx")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    writer = None
    ball_coords = []
    sticker_coords = []
    stationary_sum = np.zeros(6, dtype=float)
    stationary_count = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if h is None:
            h, w = frame.shape[:2]
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        t = frame_idx / fps
        frame_idx += 1
        results = model(frame, verbose=False)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            cx, cy, radius_px, distance = measure_ball(boxes[best_idx])
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
            cv2.circle(frame, (int(cx), int(cy)), int(radius_px), (0, 255, 0), 2)
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
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
                    color = (0, 0, 255)
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
                    color = (255, 0, 0)
                if hasattr(cv2.aruco, "drawAxis"):
                    cv2.aruco.drawAxis(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, length)
                else:
                    cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, length)
                mc = (int(corners[i][0][:, 0].mean()), int(corners[i][0][:, 1].mean()))
                cv2.putText(
                    frame,
                    f"x:{x:.2f} y:{y:.2f} z:{z:.2f}",
                    (mc[0], mc[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"r:{roll:.1f} p:{pitch:.1f} y:{yaw:.1f}",
                    (mc[0], mc[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        if writer is not None:
            writer.write(frame)
    cap.release()
    if writer is not None:
        writer.release()
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
    if writer is not None:
        print(f"Annotated video saved to {output_path}")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
    ball_path = sys.argv[2] if len(sys.argv) > 2 else "ball_coords.json"
    sticker_path = sys.argv[3] if len(sys.argv) > 3 else "sticker_coords.json"
    stationary_path = sys.argv[4] if len(sys.argv) > 4 else "stationary_sticker.json"
    output_path = sys.argv[5] if len(sys.argv) > 5 else "annotated_output.mp4"
    process_video(video_path, ball_path, sticker_path, stationary_path, output_path)
