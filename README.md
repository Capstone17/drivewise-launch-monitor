# Webcam Golf Ball Detector

This repository contains a small Python application that uses a YOLOv8
ONNX model to detect golf balls from a webcam stream. Detected golf
balls are highlighted with bounding boxes in the displayed video.

## Requirements

* Python 3.8+
* OpenCV (`opencv-python`)
* The `ultralytics` package for running YOLOv8 models
* `golf_ball_detector.onnx` (your trained model file) placed in the
  repository root

Install the required Python packages with:

```bash
pip install opencv-python ultralytics
```

## Usage

1. Place your `golf_ball_detector.onnx` file in the repository root.
2. Run the detection script:

```bash
python golf_ball_detector.py
```

Press `q` in the display window to exit the application.

## Notes

The script uses the default webcam at index `0`. If you have multiple
cameras, change the index in `cv2.VideoCapture(0)` inside
`golf_ball_detector.py`.
