import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite

# Load TFLite model with multithreading
interpreter = tflite.Interpreter(
    model_path="../golf_ball_detector.tflite",
    num_threads=4  # Use 4 cores on Raspberry Pi 5
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture("res_tst_400.mp4")

frame_count = 0
total_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (192, 128))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()

    total_time += (end - start)
    frame_count += 1
    print(f"Frames processed: {frame_count}")

cap.release()

if frame_count > 0:
    print(f"TFLite CPU Avg FPS: {frame_count / total_time:.2f}")
else:
    print("No frames processed.")
