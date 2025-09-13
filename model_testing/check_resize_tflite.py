import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path="model.tflite")
input_details = interpreter.get_input_details()

# If output is fixed: [1, 640, 640, 3], resize needed
# If not: [-1, -1, -1, 3], resizing is dynamic
print(input_details[0]['shape'])
