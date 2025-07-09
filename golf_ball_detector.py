import cv2
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from ultralytics import YOLO

# Camera parameters and real world constants
ACTUAL_BALL_RADIUS = 2.135  # centimeters

FOCAL_LENGTH = 1000.0        # pixels - approximate webcam focal length


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


def compute_speed(curr: BallMeasurement, prev: BallMeasurement):
    """Compute ball velocity vector from two measurements."""
    dt = curr.timestamp - prev.timestamp
    if dt <= 0:
        return 0.0, 0.0, 0.0
    avg_z = (curr.distance + prev.distance) / 2.0
    vx = ((curr.cx - prev.cx) / FOCAL_LENGTH) * avg_z / dt
    vy = ((curr.cy - prev.cy) / FOCAL_LENGTH) * avg_z / dt
    vz = (curr.distance - prev.distance) / dt
    return vx, vy, vz

def speed_magnitude(vx: float, vy: float, vz: float) -> float:
    """Return combined speed from its components."""
    return (vx * vx + vy * vy + vz * vz) ** 0.5

def main():
    model = YOLO('golf_ball_detector.onnx')

    # Parameters for speed waveform display
    graph_width = 300
    graph_height = 150
    history_sec = 5.0
    # Keep at most graph_width entries to bound memory usage
    speed_history = deque(maxlen=graph_width)


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

    prev_meas = None

    # Main detection loop
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Failed to grab frame from webcam.')
            break

        speed = 0.0
        vx = vy = vz = 0.0
        annotated_frame = frame
        results = model(frame)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            meas = measure_ball(boxes[best_idx])
            if prev_meas is not None:
                vx, vy, vz = compute_speed(meas, prev_meas)
            prev_meas = meas
            speed = speed_magnitude(vx, vy, vz)
            annotated_frame = results[0].plot()
            info = (
                f"Dist:{meas.distance:.2f}cm "
                f"Vx:{vx:.2f} Vy:{vy:.2f} Vz:{vz:.2f} V:{speed:.2f}"
            )
            cv2.putText(
                annotated_frame,
                info,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # Update speed waveform data
        now = time.time()
        speed_history.append((now, speed))
        while speed_history and now - speed_history[0][0] > history_sec:
            speed_history.popleft()

        graph = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
        if speed_history:
            start_time = now - history_sec
            times, points = zip(*speed_history)
            max_speed = max(max(points), 50.0)
            prev_x = prev_y = None
            for ts, sp in zip(times, points):
                x = int((ts - start_time) / history_sec * (graph_width - 1))
                y = int(graph_height - (sp / max_speed) * graph_height)
                if prev_x is not None:
                    cv2.line(graph, (prev_x, prev_y), (x, y), (0, 255, 0), 2)
                prev_x, prev_y = x, y
            cv2.putText(graph, f"Max:{max_speed:.1f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow('Webcam Golf Ball Detection', annotated_frame)
        cv2.imshow('Speed Waveform', graph)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()