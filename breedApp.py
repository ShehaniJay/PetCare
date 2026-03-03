import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="Pet Breed Classifier", page_icon="🐶")

st.title("🐶 AI Pet Breed Classifier")
st.write("Upload an image of your pet to identify its breed.")

# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_breed_model():
    return tf.keras.models.load_model("pet_breed_model.keras")

breed_model = load_breed_model()
class_names = np.load("class_names.npy", allow_pickle=True)

# ------------------------
# Image Upload
# ------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = breed_model.predict(img_array)
        confidence = np.max(prediction)
        predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Predicted Breed: {predicted_class}")
    st.info(f"Confidence: {confidence*100:.2f}%")