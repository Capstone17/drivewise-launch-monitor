import json
import sys
import time

import cv2

# Constants from ``video_ball_detector.py``
ACTUAL_BALL_RADIUS = 2.135  # centimeters
FOCAL_LENGTH = 1000.0  # pixels


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def map_to_frame(data, fps):
    mapping = {}
    for entry in data:
        idx = int(round(entry["time"] * fps))
        mapping[idx] = entry
    return mapping


def annotate_video(video_path: str, ball_json: str, sticker_json: str, output: str):
    ball_data = load_json(ball_json)
    sticker_data = load_json(sticker_json)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    ball_map = map_to_frame(ball_data, fps)
    sticker_map = map_to_frame(sticker_data, fps)

    frame_idx = 0
    ball_times = []
    sticker_times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        if frame_idx in ball_map:
            b = ball_map[frame_idx]
            distance = b["z"] + 30.0
            cx = int(b["x"] * FOCAL_LENGTH / distance + width / 2)
            cy = int(b["y"] * FOCAL_LENGTH / distance + height / 2)
            radius = int(ACTUAL_BALL_RADIUS * FOCAL_LENGTH / distance)
            cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
            cv2.putText(frame, f"t={b['time']:.2f}", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            ball_times.append((frame_idx, time.perf_counter() - start))
        if frame_idx in sticker_map:
            s = sticker_map[frame_idx]
            text = f"sticker t={s['time']:.2f} yaw={s['yaw']:.1f}"
            cv2.putText(frame, text, (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            sticker_times.append((frame_idx, time.perf_counter() - start))

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print("Ball frame processing times:")
    for idx, dt in ball_times:
        print(f"Frame {idx}: {dt:.4f}s")

    print("Sticker frame processing times:")
    for idx, dt in sticker_times:
        print(f"Frame {idx}: {dt:.4f}s")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
    ball_json = sys.argv[2] if len(sys.argv) > 2 else "ball_coords.json"
    sticker_json = sys.argv[3] if len(sys.argv) > 3 else "sticker_coords.json"
    out_path = sys.argv[4] if len(sys.argv) > 4 else "annotated.mp4"
    annotate_video(video_path, ball_json, sticker_json, out_path)
