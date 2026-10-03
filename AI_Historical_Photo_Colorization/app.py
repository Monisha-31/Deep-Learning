import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/colorization_autoencoder.keras"
IMG_SIZE = 128

st.set_page_config(
    page_title="HistoricColor AI",
    page_icon="🎨",
    layout="wide"
)

st.title("🕰️ HistoricColor AI")
st.subheader("AI-Based Historical Photo Colorization")
st.write(
    "Upload a black-and-white photograph and let a CNN Autoencoder "
    "predict a plausible color version."
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

if model is None:
    st.error(
        "Trained model not found. Run train.py first, then start the app again."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a black-and-white historical photograph",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.write("### Preview")
    st.image(image_np, caption="Uploaded photograph", use_container_width=True)

    if st.button("🎨 Colorize Image", type="primary"):
        resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))

        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        gray_input = gray.astype(np.float32) / 255.0
        gray_input = np.expand_dims(gray_input, axis=(0, -1))

        with st.spinner("AI is colorizing the photograph..."):
            prediction = model.predict(gray_input, verbose=0)[0]

        prediction = np.clip(prediction, 0, 1)
        colorized_uint8 = (prediction * 255).astype(np.uint8)

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Original / Grayscale")
            st.image(gray, clamp=True, use_container_width=True)

        with col2:
            st.write("### AI Colorized")
            st.image(colorized_uint8, use_container_width=True)

        result_bytes = cv2.imencode(
            ".png",
            cv2.cvtColor(colorized_uint8, cv2.COLOR_RGB2BGR)
        )[1].tobytes()

        st.download_button(
            "⬇️ Download Colorized Image",
            data=result_bytes,
            file_name="colorized_historical_photo.png",
            mime="image/png"
        )

        st.info(
            "The model predicts plausible colors learned from its training data; "
            "it cannot guarantee the photograph's exact original historical colors."
        )
