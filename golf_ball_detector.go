package main

import (
	"fmt"
	"image"
	"image/color"
	"time"

	ort "github.com/microsoft/onnxruntime-go"
	"gocv.io/x/gocv"
)

const (
	// Real golf ball radius in inches
	actualBallRadius = 2.135
	// Approximate focal length in pixels
	focalLength = 800.0
)

// BallMeasurement stores a detection converted to camera space.
type BallMeasurement struct {
	Timestamp float64
	CX        float64
	CY        float64
	RadiusPX  float64
	Distance  float64
}

// measureBall converts a YOLO box [x1,y1,x2,y2] to a measurement.
func measureBall(box [4]float32) BallMeasurement {
	x1, y1 := float64(box[0]), float64(box[1])
	x2, y2 := float64(box[2]), float64(box[3])
	w := x2 - x1
	h := y2 - y1
	radius := (w + h) / 4.0
	cx := (x1 + x2) / 2.0
	cy := (y1 + y2) / 2.0
	dist := focalLength * actualBallRadius / radius
	return BallMeasurement{
		Timestamp: float64(time.Now().UnixNano()) / 1e9,
		CX:        cx,
		CY:        cy,
		RadiusPX:  radius,
		Distance:  dist,
	}
}

// computeSpeed returns the velocity of the ball in camera space.
func computeSpeed(curr, prev BallMeasurement) (float64, float64, float64) {
	dt := curr.Timestamp - prev.Timestamp
	if dt <= 0 {
		return 0, 0, 0
	}
	avgZ := (curr.Distance + prev.Distance) / 2
	vx := ((curr.CX - prev.CX) / focalLength) * avgZ / dt
	vy := ((curr.CY - prev.CY) / focalLength) * avgZ / dt
	vz := (curr.Distance - prev.Distance) / dt
	return vx, vy, vz
}

func main() {
	// Open default webcam
	cam, err := gocv.OpenVideoCapture(0)
	if err != nil {
		fmt.Printf("failed to open camera: %v\n", err)
		return
	}
	defer cam.Close()

	win := gocv.NewWindow("Webcam Golf Ball Detection")
	defer win.Close()

	// Load ONNX model using onnxruntime
	session, err := ort.NewSession("golf_ball_detector.onnx")
	if err != nil {
		fmt.Printf("failed to load model: %v\n", err)
		return
	}
	defer session.Close()

	img := gocv.NewMat()
	defer img.Close()

	var prev *BallMeasurement

	for {
		if ok := cam.Read(&img); !ok || img.Empty() {
			continue
		}

		blob := gocv.BlobFromImage(img, 1.0/255.0, image.Pt(640, 640), gocv.NewScalar(0, 0, 0, 0), true, false)
		defer blob.Close()

		if err := session.SetInputTensor("images", blob); err != nil {
			fmt.Println("failed to set input tensor:", err)
			break
		}
		out, err := session.Run()
		if err != nil || len(out) == 0 {
			fmt.Println("inference error:", err)
			break
		}

		data, _ := out[0].Value().([]float32)
		// Each detection is [cx, cy, w, h, conf, class]
		var bestConf float32
		var bestBox [4]float32
		for i := 0; i+5 < len(data); i += 6 {
			conf := data[i+4]
			if conf > bestConf {
				cx := data[i]
				cy := data[i+1]
				w := data[i+2]
				h := data[i+3]
				bestBox = [4]float32{cx - w/2, cy - h/2, cx + w/2, cy + h/2}
				bestConf = conf
			}
		}

		if bestConf > 0 {
			meas := measureBall(bestBox)
			vx, vy, vz := 0.0, 0.0, 0.0
			if prev != nil {
				vx, vy, vz = computeSpeed(meas, *prev)
			}
			prev = &meas

			rect := image.Rect(int(bestBox[0]), int(bestBox[1]), int(bestBox[2]), int(bestBox[3]))
			gocv.Rectangle(&img, rect, color.RGBA{0, 255, 0, 0}, 2)
			info := fmt.Sprintf("Dist:%.2f in Vx:%.2f Vy:%.2f Vz:%.2f", meas.Distance, vx, vy, vz)
			gocv.PutText(&img, info, image.Pt(10, 30), gocv.FontHersheySimplex, 0.7, color.RGBA{0, 255, 0, 0}, 2)
		}

		win.IMShow(img)
		if win.WaitKey(1) == int('q') {
			break
		}
	}
}
