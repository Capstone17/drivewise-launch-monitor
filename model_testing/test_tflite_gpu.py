import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate

# Load with GPU delegate
interpreter = tflite.Interpreter(
    model_path="model.tflite",
    experimental_delegates=[load_delegate('libedgetpu.so.1')]  # for Coral EdgeTPU
    # OR for RPi OpenGL ES GPU delegate:
    # experimental_delegates=[load_delegate('libtensorflowlite_gpu_delegate.so')]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture("input.mp4")

frame_count = 0
total_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()

    total_time += (end - start)
    frame_count += 1

cap.release()
print(f"TFLite GPU Avg FPS: {frame_count / total_time:.2f}")
