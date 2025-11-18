from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np
import video_ball_detector as vbd

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "golf_ball_detector.tflite")

@dataclass(slots=True)
class BallMeasurement:
    bbox: tuple[float, float, float, float]
    score: float
    center: tuple[float, float]
    radius: float
    distance: float
    y_position: float


def _select_best_detection(
    frame_shape: Tuple[int, int, int],
    detections: list[dict],
) -> BallMeasurement | None:
    if not detections:
        return None
    height, width = frame_shape[:2]
    edge_safe = [det for det in detections if vbd.bbox_within_image(det["bbox"], width, height)]
    candidates = edge_safe if edge_safe else detections
    best = max(candidates, key=lambda d: d["score"])

    if best["score"] < vbd.BALL_SCORE_THRESHOLD:
        return None

    x1, y1, x2, y2 = best["bbox"]
    cx, cy, radius, distance = vbd.bbox_to_ball_metrics(x1, y1, x2, y2)
    if radius < vbd.MIN_BALL_RADIUS_PX:
        return None

    y_offset = cy - height / 2.0
    y_position = y_offset * distance / vbd.FOCAL_LENGTH
    return BallMeasurement(
        bbox=(x1, y1, x2, y2),
        score=float(best["score"]),
        center=(cx, cy),
        radius=radius,
        distance=distance,
        y_position=y_position,
    )


def detect_ball_in_frame(
    frame: np.ndarray,
    *,
    detector: vbd.TFLiteBallDetector | None = None,
    calibration: dict[str, object] | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
) -> tuple[BallMeasurement | None, np.ndarray]:

    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty")

    if calibration is not None:
        vbd.apply_calibration(calibration)

    owned_detector = detector is None
    if detector is None:
        detector = vbd.TFLiteBallDetector(model_path, conf_threshold=0.01)

    enhanced, _ = vbd.preprocess_frame(frame)
    detections = detector.detect(enhanced)
    measurement = _select_best_detection(enhanced.shape, detections)

    if owned_detector and hasattr(detector, "interpreter"):
        del detector

    return measurement, enhanced


def find_ball_y_in_image(
    image_path: str,
    *,
    detector: vbd.TFLiteBallDetector | None = None,
    calibration: dict[str, object] | None = None,
) -> float:

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    measurement, _ = detect_ball_in_frame(frame, detector=detector, calibration=calibration)
    if measurement is None:
        raise RuntimeError("Ball not detected in the provided image")

    return float(measurement.y_position)


def _load_json(path: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Report the Y coordinate of the golf ball from an image.")
    parser.add_argument("image_path", help="Path to the JPEG image.")
    parser.add_argument(
        "--calibration",
        help="Optional path to a JSON file with calibration overrides.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the TFLite model (defaults to golf_ball_detector.tflite in the repo root).",
    )
    args = parser.parse_args()

    calibration_data = _load_json(args.calibration) if args.calibration else None
    detector = vbd.TFLiteBallDetector(args.model, conf_threshold=0.01)
    try:
        y_coord = find_ball_y_in_image(
            args.image_path,
            detector=detector,
            calibration=calibration_data,
        )
    finally:
        del detector
    print(f"Ball Y position: {y_coord:.2f}")
