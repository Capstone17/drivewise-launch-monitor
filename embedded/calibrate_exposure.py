from picamera2 import Picamera2
import numpy as np
import cv2

picam2 = Picamera2()

# Configure camera for RGB output
config = picam2.create_still_configuration(main={"format": "XRGB8888"})
picam2.configure(config)

# Set manual exposure
picam2.set_controls({"AeEnable": False, "ExposureTime": 10000, "AnalogueGain": 1.0, "DigitalGain": 1.0})

picam2.start()
frame = picam2.capture_array()
picam2.stop()

# Save image
cv2.imwrite("manual_exposure.jpg", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
print("Saved image with manual exposure.")
