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

## Go Implementation

For better performance a Go version of the detector is included. It
relies on `gocv` for video capture and `onnxruntime-go` for inference.

### Additional Requirements

* Go 1.18+
* `gocv` and `onnxruntime-go` libraries

### Run the Go detector

```bash
go run golf_ball_detector.go
```

The Go application mirrors the Python features, estimating distance and
3‑D velocity of the detected golf ball.
