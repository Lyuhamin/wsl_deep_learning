from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# 모델 로드
model_path = "my_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
model = tf.keras.models.load_model(model_path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    img_file = request.files["image"].read()
    img = Image.open(io.BytesIO(img_file)).resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    class_indices = {
        "메카인": 0,
        "밴드": 1,
        "베아제": 2,
        "알러샷": 3,
        "제감콜드": 4,
        "타이레놀": 5,
        "판콜A": 6,
        "판피린": 7,
        "후시딘": 8,
        "훼스탈": 9,
    }
    class_names = {v: k for k, v in class_indices.items()}

    return jsonify({"predicted_class": class_names[predicted_class]})
