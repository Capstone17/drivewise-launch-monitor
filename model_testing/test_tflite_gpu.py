import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate

# ------------------------------
# Load TFLite model with GPU delegate
# ------------------------------
try:
    # RPi GPU delegate (OpenGL ES)
    delegate = load_delegate('libtensorflowlite_gpu_delegate.so')
    interpreter = tflite.Interpreter(model_path="../golf_ball_detector.tflite",
                                     experimental_delegates=[delegate])
except ValueError:
    print("GPU delegate not found, falling back to CPU")
    interpreter = tflite.Interpreter(model_path="../golf_ball_detector.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']
scale, zero_point = input_details[0]['quantization']

# ------------------------------
# Letterbox function
# ------------------------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_shape[1], new_shape[0], 3), color, dtype=np.uint8)
    top, left = (new_shape[1] - nh) // 2, (new_shape[0] - nw) // 2
    canvas[top:top+nh, left:left+nw, :] = img_resized
    return canvas

# ------------------------------
# Open video
# ------------------------------
cap = cv2.VideoCapture("res_tst_400.mp4")
frame_count = 0
total_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = letterbox(frame, new_shape=(640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)

    # Handle dtype
    if input_dtype == np.float32:
        img = img.astype(np.float32) / 255.0
    elif input_dtype == np.int8:
        img = img.astype(np.float32) / 255.0
        img = (img / scale + zero_point).astype(np.int8)
    else:
        raise ValueError(f"Unsupported input dtype: {input_dtype}")

    # Inference
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()

    total_time += (end - start)
    frame_count += 1
    print(f"Frames processed: {frame_count}")

cap.release()
print(f"TFLite GPU Avg FPS: {frame_count / total_time:.2f}")
