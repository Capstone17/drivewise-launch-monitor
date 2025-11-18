from __future__ import annotations

import argparse
import json
import os

import cv2

import video_ball_detector as vbd

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "golf_ball_detector.tflite")


def find_ball_y_in_image(
    image_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    calibration: dict[str, object] | None = None,
) -> float:
    """Return the golf ball's bottom Y coordinate in inches, matching ``video_ball_detector``."""

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    if calibration is not None:
        vbd.apply_calibration(calibration)

    detector = vbd.TFLiteBallDetector(model_path, conf_threshold=0.01)
    try:
        enhanced, _ = vbd.preprocess_frame(frame)
        detections = detector.detect(enhanced)
    finally:
        del detector

    if not detections:
        raise RuntimeError("Ball not detected in the image")

    best = max(detections, key=lambda d: d["score"])
    x1, y1, x2, y2 = best["bbox"]
    cx, cy, radius, distance = vbd.bbox_to_ball_metrics(x1, y1, x2, y2)
    if radius < vbd.MIN_BALL_RADIUS_PX:
        raise RuntimeError("Detection too small to be a valid golf ball")

    h = frame.shape[0]
    bottom_pixel_offset = max(y1, y2) - h / 2.0
    bottom_inches = bottom_pixel_offset * distance / vbd.FOCAL_LENGTH
    return float(bottom_inches)


def _load_json(path: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the calibrated bottom Y coordinate of the golf ball in inches.")
    parser.add_argument("image_path", help="Path to the input JPEG.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the golf_ball_detector.tflite model.",
    )
    parser.add_argument(
        "--calibration",
        help="Optional JSON file with calibration overrides.",
    )
    args = parser.parse_args()

    calib = _load_json(args.calibration) if args.calibration else None
    y = find_ball_y_in_image(args.image_path, args.model, calib)
    print(f"Ball bottom Y (in): {y:.2f}")


if __name__ == "__main__":
    main()
