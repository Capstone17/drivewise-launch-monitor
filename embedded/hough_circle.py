import cv2
import numpy as np
import os

# === LOAD CAMERA CALIBRATION ===
_calib_path = os.path.join(os.path.dirname(__file__), "..", "calibration", "camera_calib.npz")
_calib_data = np.load(_calib_path)
CAMERA_MATRIX = _calib_data["K"]
FOCAL_LENGTH = (CAMERA_MATRIX[0, 0] + CAMERA_MATRIX[1, 1]) / 2.0
REAL_RADIUS = 21.35    # real radius in mm (e.g., for a golf ball ~21.35 mm)

# Load the video
video_path = 'tst.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Grayscale + blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=35,
        param1=100,
        param2=50,
        minRadius=20,
        maxRadius=50
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Use the first (clearest) circle
        x, y, r = circles[0]

        # Calculate distance from camera
        distance_mm = (FOCAL_LENGTH * REAL_RADIUS) / r

        # Draw the circle and distance
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
        cv2.putText(frame,
                    f"Distance: {distance_mm:.1f} mm",
                    (x - 60, y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)

    # Show result
    cv2.imshow("Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
