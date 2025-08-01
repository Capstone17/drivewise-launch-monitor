# Step 1: Print and prepare a calibration pattern
#   See checkerboard.pdf in this folder, which has 9x6 inner corners and 23mm square length
#   Verify its exact size in mm and update square_size below
# Step 2: Capture calibration images
#   Take 10-20 images of the pattern at different angles and distances
#   Example command: libcamera-still -o img%02d.jpg --width 1440 --height 1080 -n
# Step 3: Run the script below
# Step 4: Outputs
#   K[0,0], K[1,1]:	fx, fy (focal length in pixels)
#   K[0,2], K[1,2]:	cx, cy (principal point in pixels)
#   dist:	Distortion coefficients: [k1, k2, p1, p2, k3]
#   rvecs, tvecs:	Rotation/translation of the board in each view


import cv2
import numpy as np
import glob

# Settings
CHECKERBOARD = (9, 6)  # inner corners
square_size = 23.0     # mm

# 3D points (world coordinates)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store 3D and 2D points
objpoints = []  # 3D
imgpoints = []  # 2D

images = glob.glob('img*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# Calibration
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix:\n", K)
print("Focal lengths (fx, fy):", K[0, 0], K[1, 1])
print("Principal point (cx, cy):", K[0, 2], K[1, 2])
print("Distortion coefficients:", dist.ravel())

# Save to file
np.savez("camera_calib.npz", K=K, dist=dist)