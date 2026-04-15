import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_DIR = "models/plant_savedmodel"

model = tf.keras.models.load_model(MODEL_DIR, compile=False)

with open("class_indices.json", "r") as f:
    labels = json.load(f)

classes = {v: k for k, v in labels.items()}

def predict_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]

    idx = np.argmax(pred)
    confidence = round(float(np.max(pred)) * 100, 2)

    return classes[idx], confidence
