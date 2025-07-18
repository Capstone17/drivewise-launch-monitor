## Requirements

* Python 3.8+
* `imageio` for reading video files
* The `ultralytics` package for running YOLOv8 models
* `golf_ball_detector.onnx` (your trained model file) placed in the
  repository root

Install the required Python packages with:

```bash
pip install imageio ultralytics
```

## Usage

1. Place your `golf_ball_detector.onnx` file in the repository root and your
   MP4 video (e.g. `video.mp4`).
2. Run the detection script:

```bash
python video_ball_detector.py video.mp4 ball_coords.json
```

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
