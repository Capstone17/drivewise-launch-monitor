import cv2
import cv2.aruco as aruco
import numpy as np

# Select dictionary
# Using the smallest possible dictionary to account for noise and blur (50)
# Using a custom dictionary is also possible for maximum error correction capability
# dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)  # Best option
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)  # For testing purposes only

# Load image
frame = cv2.imread("img.png")

# Detect markers
corners, ids, _ = aruco.detectMarkers(frame, dictionary)

# Draw detected markers
frame_marked = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
cv2.imshow("Detected Markers", frame_marked)
cv2.waitKey(0)

# Load calibration parameters
data = np.load("../calibration/camera_calib.npz")
K = data['K']
dist = data['dist']

# Specify lengths
marker_length = 0.03  # m
axis_length = 0.015  # m

# Camera Matrix: K
rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, K, dist)
print("rvecs shape:", rvec.shape)   # Should be (N, 1, 3)
print("tvecs shape:", tvec.shape)   # Should be (N, 1, 3)
print("rvec:\n", rvec)
print("tvec:\n", tvec)

# Draw axes
for i in range(len(ids)):
    cv2.drawFrameAxes(frame, K, dist, rvec[i], tvec[i], axis_length)

# Show the final frame
cv2.imshow("Pose Estimation", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Example: tvec from ArUco pose estimation (shape: (1, 1, 3))
m_x, m_y, m_z = tvec[0][0]  # in meters
cm_x, cm_y, cm_z = tvec[0][0] * 100  # in centimeters

print(f"Marker position (m):  x={m_x:.3f} m, y={m_y:.3f} m, z={m_z:.3f} m")
print(f"Marker position (cm): x={cm_x:.1f} cm, y={cm_y:.1f} cm, z={cm_z:.1f} cm")