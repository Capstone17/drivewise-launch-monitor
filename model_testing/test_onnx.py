import cv2
import onnxruntime as ort
import numpy as np
import time

# Load ONNX model
session = ort.InferenceSession("../golf_ball_detector.onnx", providers=["CPUExecutionProvider"])

# Open video
cap = cv2.VideoCapture("res_tst_400.mp4")

frame_count = 0
total_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess (adjust if your model needs resize/normalize)
    img = cv2.resize(frame, (192, 128))  
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)   # Add batch dim

    start = time.time()
    outputs = session.run(None, {session.get_inputs()[0].name: img})
    end = time.time()

    total_time += (end - start)
    frame_count += 1
    print(f"Frames processed: {frame_count}")

cap.release()
print(f"ONNX Avg FPS: {frame_count / total_time:.2f}")
