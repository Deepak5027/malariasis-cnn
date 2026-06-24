import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Load model
model_path = os.path.join("models", "CNN.h5")
model = load_model(model_path)

classes = {
    0: "Parasitized",
    1: "Uninfected"
}

# UI
st.title("🦠 Malaria Detection System")
st.write("Upload a cell image to detect infection")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Convert file to numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = cv2.resize(img, (50, 50))
    img_array = img_resized.reshape(1, 50, 50, 3)
    img_array = img_array / 255.0

    # Prediction
    pred = model.predict(img_array)
    index = int(np.argmax(pred[0]))
    result = classes[index]

    st.subheader("Prediction Result:")
    st.success(result)
