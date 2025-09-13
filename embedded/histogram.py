#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt

# ----------------------------
# Load image
# ----------------------------
image_path = "test.jpg"  # replace with your filename
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Compute histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# ----------------------------
# Plot histogram
# ----------------------------
plt.figure(figsize=(8,4))
plt.title("Grayscale Histogram")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")
plt.plot(hist, color="black")
plt.xlim([0, 255])
plt.show()
