from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "models", "CNN.h5")
model = load_model(model_path)

classes = {
    0: "Parasitized",
    1: "Uninfected"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    os.makedirs("uploads", exist_ok=True)

    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)

    if img is None:
        return "Invalid image"

    img = cv2.resize(img, (50, 50))
    img = img.reshape(-1, 50, 50, 3)
    img = img / 255.0

    pred = model.predict(img, verbose=0)

    index = int(np.argmax(pred[0]))

    prediction = classes[index]

    return render_template(
        "result.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
