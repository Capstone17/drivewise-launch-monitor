import cv2
import numpy as np

# Load your image
image = cv2.imread('frame_0751.png')  # replace with your image path
output = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply median blur to reduce noise
gray = cv2.medianBlur(gray, 5)

# Detect circles using HoughCircles with tuned parameters for golf ball size
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=50,
    param1=50,
    param2=40,       # Higher threshold to reduce false positives
    minRadius=0,    # Adjust based on expected golf ball size in pixels
    maxRadius=60
)

# If circles detected, select only the best one (strongest or largest)
if circles is not None:
    circles = np.uint16(np.around(circles))
    # Option 1: Take the first detected circle
    # best_circle = circles[0, 0]

    # Option 2: Take the circle with the largest radius (more robust)
    best_circle = max(circles[0, :], key=lambda x: x[2])

    # Draw the outer circle
    cv2.circle(output, (best_circle[0], best_circle[1]), best_circle[2], (0, 255, 0), 2)
    # Draw the center of the circle
    cv2.circle(output, (best_circle[0], best_circle[1]), 2, (0, 0, 255), 3)

# Display the result
cv2.imshow('Detected Golf Ball', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
