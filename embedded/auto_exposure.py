#!/usr/bin/env python3
from picamera2 import Picamera2, Preview
import time
import cv2
import numpy as np

# ----------------------------
# Configuration
# ----------------------------
AE_TEST_DURATION = 5.0  # seconds to let AE stabilize
FRAME_INTERVAL = 0.5    # seconds between frames

# ----------------------------
# Initialize Picamera2
# ----------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "XRGB8888"})
picam2.configure(config)

# Enable Auto Exposure
picam2.set_controls({"AeEnable": True})

# Start camera
picam2.start()
print("Camera started. Auto Exposure enabled.")

start_time = time.time()
frame_count = 0

try:
    while time.time() - start_time < AE_TEST_DURATION:
        # Capture frame as numpy array
        frame = picam2.capture_array()
        frame_count += 1

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()

        # Compute mean brightness
        mean_brightness = gray.mean()

        # Get AE metadata (exposure time, analog gain, digital gain)
        metadata = picam2.capture_metadata()
        exposure_us = metadata.get("ExposureTime", 0)
        analog_gain = metadata.get("AnalogueGain", 0)
        digital_gain = metadata.get("DigitalGain", 0)

        # Print info
        print(f"Frame {frame_count}: Exposure={exposure_us} us, "
              f"AnalogGain={analog_gain:.2f}, DigitalGain={digital_gain:.2f}, "
              f"MeanBrightness={mean_brightness:.1f}")

        time.sleep(FRAME_INTERVAL)

finally:
    picam2.stop()
    print("Camera stopped.")

