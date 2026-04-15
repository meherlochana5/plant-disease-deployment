import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="plant_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("class_indices.json", "r") as f:
    labels = json.load(f)

classes = {v: k for k, v in labels.items()}

def predict_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]["index"])[0]

    idx = np.argmax(pred)
    confidence = round(float(np.max(pred)) * 100, 2)

    return classes[idx], confidence
