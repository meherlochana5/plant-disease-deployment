import tensorflow as tf
import numpy as np
import json
import os
import gdown
import zipfile
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_DIR = "models/plant_savedmodel"
ZIP_PATH = "models/plant_savedmodel.zip"
FILE_ID = "1dJLrhlVVs7GjvWi1SKRsxiycC97wrAEt"

os.makedirs("models", exist_ok=True)

# Download and extract model
if not os.path.exists(MODEL_DIR):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, ZIP_PATH, quiet=False)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("models")

# Load SavedModel directly
model = tf.saved_model.load(MODEL_DIR)
infer = model.signatures["serving_default"]

# Labels
with open("class_indices.json", "r") as f:
    labels = json.load(f)

classes = {v: k for k, v in labels.items()}

def predict_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    pred = infer(tf.constant(img))
    pred = list(pred.values())[0].numpy()[0]

    idx = np.argmax(pred)
    confidence = round(float(np.max(pred)) * 100, 2)

    return classes[idx], confidence
