from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

model = load_model("models/cnn.h5")

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
    file = request.files["image"]

    os.makedirs("uploads", exist_ok=True)

    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    img = cv2.resize(img, (50,50))
    img = img.reshape(-1,50,50,3)
    img = img / 255.0

    pred = model.predict(img)
    index = pred.argmax()

    result = classes[index]

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
