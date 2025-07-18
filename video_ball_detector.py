import imageio.v2 as imageio
from ultralytics import YOLO
import numpy as np
import json
import sys

ACTUAL_BALL_RADIUS = 2.135  # centimeters
FOCAL_LENGTH = 1000.0        # pixels


def measure_ball(box):
    x1, y1, x2, y2 = box.xyxy[0]
    w = float(x2 - x1)
    h = float(y2 - y1)
    radius_px = (w + h) / 4.0
    cx = float(x1 + x2) / 2.0
    cy = float(y1 + y2) / 2.0
    distance = FOCAL_LENGTH * ACTUAL_BALL_RADIUS / radius_px
    return cx, cy, radius_px, distance


def process_video(video_path, output_path):
    model = YOLO('golf_ball_detector.onnx')
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data().get('fps', 30)
    w = h = None
    coords = []

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
            coords.append([t, [round(bx, 2), round(by, 2), round(bz, 2)]])

    reader.close()
    with open(output_path, 'w') as f:
        json.dump(coords, f)
    print(f'Saved {len(coords)} points to {output_path}')


if __name__ == '__main__':
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'video.mp4'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'ball_coords.json'
    process_video(video_path, output_path)
