# Step 1: Print and prepare a calibration pattern
#   See checkerboard.pdf in this folder, which has 9x6 inner corners and 23mm square length
#   Verify its exact size in mm and update square_size below
# Step 2: Capture calibration images
#   Take 10-20 images of the pattern at different angles and distances
#   Example command: libcamera-still -o img%02d.jpg --width 1440 --height 1080 -n
# Step 3: Input cropping parameters
# Step 4: Run the script below
# Step 5: Outputs
#   K[0,0], K[1,1]:	fx, fy (focal length in pixels)
#   K[0,2], K[1,2]:	cx, cy (principal point in pixels)
#   dist:	Distortion coefficients: [k1, k2, p1, p2, k3]
#   rvecs, tvecs:	Rotation/translation of the board in each view


import cv2
import numpy as np
import glob
from crop_calibration import adjust_camera_matrix

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
print("====BEFORE CROPPING====")
print("Camera matrix:\n", K)
print("Focal lengths (fx, fy):", K[0, 0], K[1, 1])
print("Principal point (cx, cy):", K[0, 2], K[1, 2])
print("Distortion coefficients:", dist.ravel())

# Adjust for cropping
crop_x_offset = (1456 - 400) // 2  # = 312
crop_y_offset = (1088 - 144) // 2  # = 472
K_cropped = adjust_camera_matrix(K,
                                 crop_offset=(crop_x_offset, crop_y_offset),          # crop 100px left, 50px top
                                 new_size=None,                   # no resize
                                 original_size=(1456, 1080))       # full-res size from calibration
print("====AFTER CROPPING====")
print("Camera matrix:\n", K_cropped)
print("Focal lengths (fx, fy):", K_cropped[0, 0], K_cropped[1, 1])
print("Principal point (cx, cy):", K_cropped[0, 2], K_cropped[1, 2])
print("Distortion coefficients:", dist.ravel())

# Save to file
np.savez("camera_calib.npz", camera_matrix=K_cropped, dist_coeffs=dist)