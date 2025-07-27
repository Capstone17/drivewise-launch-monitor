## Requirements

* Python 3.8+
* `imageio` for reading video files
* `opencv-python` for ArUco marker detection
* The `ultralytics` package for running YOLOv8 models
* `golf_ball_detector.onnx` (your trained model file) placed in the
  repository root

Install the required Python packages with:

```bash
pip install imageio opencv-python ultralytics
```

## Usage

1. Place your `golf_ball_detector.onnx` file in the repository root and your
   MP4 video (e.g. `video.mp4`).
2. Run the detection script:

```bash
python video_ball_detector.py video.mp4 \
    ball_coords.json sticker_coords.json stationary_sticker.json
```

The first JSON file will contain ball coordinates. The second stores the
position and orientation of the moving sticker while the third contains the
average pose of the stationary sticker, for example:

```json
[
  {"time": 1.72, "x": 0.66, "y": 6.17, "z": 11.66},
  {"time": 1.76, "x": 0.59, "y": 5.75, "z": 11.26}
]
```

### Hybrid ball tracking

The ball tracker first searches with YOLOv8 until the same object is detected
for several consecutive frames. Once confirmed, a lightweight Hough circle
transform follows the ball, restricted to the previous location. YOLO is only
invoked again when the circle tracker loses the ball or at long intervals to
verify tracking. This decision tree keeps inference time low while avoiding
false positives from other circular objects.

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
