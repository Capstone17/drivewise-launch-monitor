## Requirements

* Python 3.8+
* `imageio` for reading video files
* `opencv-python` for ArUco marker detection
* OpenCV's Hough circle transform is used for ball detection so no
  additional model files are required

Install the required Python packages with:

```bash
pip install imageio opencv-python
```

## Usage

1. Place your MP4 video (e.g. `video.mp4`) in the repository root.
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
