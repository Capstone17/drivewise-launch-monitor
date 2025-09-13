import onnxruntime as ort

# Load session
sess = ort.InferenceSession("../golf_ball_detector.onnx")

# Check providers
#   If the output is something like:
#     Available providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
#     Using provider:  ['CPUExecutionProvider']
#   Then the model is running on the CPU execution backend (BAD)
print("Available providers:", ort.get_available_providers())
print("Using provider: ", sess.get_providers())
