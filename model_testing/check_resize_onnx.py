import onnxruntime as ort
session = ort.InferenceSession("../golf_ball_detector.onnx")

# If output is fixed: [1, 3, 640, 640], resize is required
# If not: [1, 3, -1, -1], resizing is dynamic
print(session.get_inputs()[0].shape)
