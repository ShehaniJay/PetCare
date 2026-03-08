import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ------------------------
# Page Setup
# ------------------------
st.set_page_config(page_title="AI Pet Health Assistant", page_icon="🐾")

st.title("🐾 AI Pet Health Assistant")
st.sidebar.title("Navigation")

menu = st.sidebar.selectbox("Choose Feature", [
    "Breed Identification",
    "Disease Prediction"
])

# ------------------------
# Load Models
# ------------------------
import tensorflow as tf
from tensorflow.keras.models import load_model

@st.cache_resource
def load_breed_model():
    return load_model("pet_breed_model.keras")

@st.cache_resource
def load_disease_model():
    model = joblib.load("disease_predictor.pkl")
    le = joblib.load("label_encoder.pkl")
    cols = joblib.load("model_columns.pkl")
    return model, le, cols

breed_model = load_breed_model()
class_names = np.load("class_names.npy", allow_pickle=True)

disease_model, le, model_columns = load_disease_model()

# =========================
# 🐶 BREED IDENTIFICATION
# =========================
if menu == "Breed Identification":

    st.header("Upload Pet Image")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing..."):
            prediction = breed_model.predict(img_array)
            confidence = np.max(prediction)
            predicted_class = class_names[np.argmax(prediction)]

        st.success(f"Predicted Breed: {predicted_class}")
        st.info(f"Confidence: {confidence*100:.2f}%")

# =========================
# 🩺 DISEASE PREDICTION
# =========================
elif menu == "Disease Prediction":

    st.header("Pet Health Disease Prediction")

    # Input features
    age = st.number_input("Age (years)", min_value=0.0)
    weight = st.number_input("Weight (kg)", min_value=0.0)
    temperature = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0)

    vomiting = st.selectbox("Vomiting?", ["No", "Yes"])
    lethargy = st.selectbox("Lethargy?", ["No", "Yes"])
    appetite_loss = st.selectbox("Loss of Appetite?", ["No", "Yes"])
    skin_lesions = st.selectbox("Skin Lesions?", ["No", "Yes"])
    breathing_difficulty = st.selectbox("Breathing Difficulty?", ["No", "Yes"])
    joint_pain = st.selectbox("Joint Pain?", ["No", "Yes"])
    breed = st.selectbox("Breed", ["Labrador", "German Shepherd", "Bulldog", "Poodle",
                                   "Beagle", "Persian Cat", "Siamese Cat", "Golden Retriever",
                                   "Rottweiler", "Maine Coon"])

    if st.button("Predict Disease"):

        # Convert categorical inputs to numeric
        features = {
            "age_years": age,
            "weight_kg": weight,
            "temperature_c": temperature,
            "vomiting": 1 if vomiting=="Yes" else 0,
            "lethargy": 1 if lethargy=="Yes" else 0,
            "appetite_loss": 1 if appetite_loss=="Yes" else 0,
            "skin_lesions": 1 if skin_lesions=="Yes" else 0,
            "breathing_difficulty": 1 if breathing_difficulty=="Yes" else 0,
            "joint_pain": 1 if joint_pain=="Yes" else 0
        }

        # One-hot encode breed
        for b in ["breed_Labrador","breed_German Shepherd","breed_Bulldog",
                  "breed_Poodle","breed_Beagle","breed_Persian Cat",
                  "breed_Siamese Cat","breed_Golden Retriever","breed_Rottweiler",
                  "breed_Maine Coon"]:
            features[b] = 1 if b.split("_")[1] == breed else 0

        # Arrange columns in model order
        input_df = pd.DataFrame([features])
        input_df = input_df[model_columns]

        # Predict
        pred_class_num = disease_model.predict(input_df)[0]
        probability = np.max(disease_model.predict_proba(input_df))

        pred_class_label = le.inverse_transform([pred_class_num])[0]

        st.success(f"Predicted Condition: {pred_class_label}")
        st.info(f"Confidence: {probability*100:.2f}%")
        st.warning("⚠️ This is an AI-based prediction. Please consult a veterinarian for confirmation.")