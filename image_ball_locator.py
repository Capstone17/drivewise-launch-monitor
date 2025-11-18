from __future__ import annotations

import argparse
import os

import cv2

import video_ball_detector as vbd

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "golf_ball_detector.tflite")


def find_ball_y_in_image(image_path: str, model_path: str = DEFAULT_MODEL_PATH) -> tuple[float, float]:
    """Return (pixel_from_top, pixels_from_bottom) for the detected ball bottom."""

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    detector = vbd.TFLiteBallDetector(model_path, conf_threshold=0.01)
    try:
        enhanced, _ = vbd.preprocess_frame(frame)
        detections = detector.detect(enhanced)
    finally:
        del detector

    if not detections:
        raise RuntimeError("Ball not detected in the image")

    best = max(detections, key=lambda d: d["score"])
    _, y1, _, y2 = best["bbox"]
    bottom_from_top = float(max(y1, y2))
    h = frame.shape[0]
    bottom_from_bottom = float(h - 1 - bottom_from_top)
    return bottom_from_top, bottom_from_bottom


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the ball bottom pixel offsets (top and bottom).")
    parser.add_argument("image_path")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    px_top, px_bottom = find_ball_y_in_image(args.image_path, args.model)
    print(f"Bottom from top: {px_top:.1f} px")
    print(f"Bottom from bottom: {px_bottom:.1f} px")


if __name__ == "__main__":
    main()
