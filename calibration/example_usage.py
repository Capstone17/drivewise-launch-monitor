import cv2
import numpy as np

# Load calibration parameters
data = np.load("camera_calib.npz")
K = data['K']
dist = data['dist']


# ArUco pose estimation: retval, rvec, tvec = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, K, dist)
