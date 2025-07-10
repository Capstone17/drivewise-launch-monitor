import cv2
import time
from dataclasses import dataclass
from ultralytics import YOLO
import numpy as np

# Camera parameters and real world constants
ACTUAL_BALL_RADIUS = 2.135  # centimeters

FOCAL_LENGTH = 800.0        # pixels - approximate webcam focal length

# ArUco parameters
MARKER_SIZE = 5.0  # centimeters, length of one side of the marker


@dataclass
class BallMeasurement:
    """Stores a single measurement of the ball."""
    timestamp: float
    cx: float
    cy: float
    radius_px: float
    distance: float


def measure_ball(box) -> BallMeasurement:
    """Convert a YOLO bounding box to a BallMeasurement."""
    x1, y1, x2, y2 = box.xyxy[0]
    w = float(x2 - x1)
    h = float(y2 - y1)
    radius_px = (w + h) / 4.0  # average half width and half height
    cx = float(x1 + x2) / 2.0
    cy = float(y1 + y2) / 2.0
    distance = FOCAL_LENGTH * ACTUAL_BALL_RADIUS / radius_px
    return BallMeasurement(time.time(), cx, cy, radius_px, distance)



def main():
    model = YOLO('golf_ball_detector.onnx')


    # Scan for a working camera index
    def find_working_camera(max_index=5):
        for idx in range(max_index):
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
                if backend is not None:
                    cap = cv2.VideoCapture(idx, backend)
                else:
                    cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    # Try to grab a frame
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    for _ in range(5):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.mean() > 5:
                            print(f"Camera found at index {idx} with backend {backend if backend is not None else 'default'}.")
                            return cap
                    cap.release()
        print("No working camera found. Tried all indices and backends.")
        return None

    cap = find_working_camera(5)
    if cap is None or not cap.isOpened():
        print('Could not open any webcam. Try closing other apps that use the camera, check permissions, or try a different camera.')
        return


    # Test: Try to grab a frame and check if it's black, fallback to lower resolution if needed
    ret, frame = cap.read()
    if not ret or frame is None:
        print('Failed to grab initial frame from webcam.')
        cap.release()
        cv2.destroyAllWindows()
        return
    if frame.mean() < 5:
        print('First frame is all black, trying lower resolution (640x480)...')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 5:
                print('Successfully got a non-black frame at 640x480.')
                break
        else:
            print('Still getting black frames. Please check your camera index, close other apps, or try a different webcam.')
            cap.release()
            cv2.destroyAllWindows()
            return

    # Show the first frame for confirmation
    cv2.imshow('Webcam Test Frame', frame)
    print(f'Frame shape: {frame.shape}, mean pixel value: {frame.mean():.2f}')
    cv2.waitKey(1000)  # Show for 1 second
    cv2.destroyWindow('Webcam Test Frame')

    # Setup ArUco detection
    h, w = frame.shape[:2]
    camera_matrix = np.array(
        [[FOCAL_LENGTH, 0, w / 2.0], [0, FOCAL_LENGTH, h / 2.0], [0, 0, 1]],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        aruco_params = cv2.aruco.DetectorParameters_create()
    else:  # fallback for newer OpenCV where constructor is used directly
        aruco_params = cv2.aruco.DetectorParameters()

    # Some OpenCV versions expose ArUco detection via the ArucoDetector class
    if hasattr(cv2.aruco, "ArucoDetector"):
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    else:
        aruco_detector = None

    # Main detection loop
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Failed to grab frame from webcam.')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if aruco_detector is not None:
            corners, ids, _ = aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        results = model(frame)
        annotated_frame = frame.copy()

        sticker_positions = []
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, camera_matrix, dist_coeffs
            )
            for i, tvec in enumerate(tvecs):
                cv2.aruco.drawAxis(
                    annotated_frame, camera_matrix, dist_coeffs, rvecs[i], tvec, MARKER_SIZE / 2
                )
                sticker_positions.append((tvec[0][0], tvec[0][1], tvec[0][2] - 30.0))

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            meas = measure_ball(boxes[best_idx])

            annotated_frame = results[0].plot()

            # Ball position relative to 30 cm origin
            bx = (meas.cx - w / 2.0) * meas.distance / FOCAL_LENGTH
            by = (meas.cy - h / 2.0) * meas.distance / FOCAL_LENGTH
            bz = meas.distance - 30.0
            ball_pos = (bx, by, bz)

            info = f"Ball:{[round(v,2) for v in ball_pos]}"
            for idx, pos in enumerate(sticker_positions):
                info += f" Sticker{idx+1}:{[round(v,2) for v in pos]}"
            cv2.putText(
                annotated_frame,
                info,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            print(info)
        else:
            annotated_frame = frame

        cv2.imshow('Webcam Golf Ball Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
