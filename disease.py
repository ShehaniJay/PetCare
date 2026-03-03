import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="Pet Disease Predictor", page_icon="🩺")

st.title("🩺 AI Pet Disease Prediction")
st.write("Enter your pet's health details to predict possible conditions.")

# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_disease_model():
    model = joblib.load("disease_predictor.pkl")
    le = joblib.load("label_encoder.pkl")
    cols = joblib.load("model_columns.pkl")
    return model, le, cols

disease_model, le, model_columns = load_disease_model()

# ------------------------
# User Inputs
# ------------------------
age = st.number_input("Age (years)", min_value=0.0)
weight = st.number_input("Weight (kg)", min_value=0.0)
temperature = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0)

vomiting = st.selectbox("Vomiting?", ["No", "Yes"])
lethargy = st.selectbox("Lethargy?", ["No", "Yes"])
appetite_loss = st.selectbox("Loss of Appetite?", ["No", "Yes"])
skin_lesions = st.selectbox("Skin Lesions?", ["No", "Yes"])
breathing_difficulty = st.selectbox("Breathing Difficulty?", ["No", "Yes"])
joint_pain = st.selectbox("Joint Pain?", ["No", "Yes"])

breed = st.selectbox("Breed", [
    "Labrador", "German Shepherd", "Bulldog", "Poodle",
    "Beagle", "Persian Cat", "Siamese Cat",
    "Golden Retriever", "Rottweiler", "Maine Coon"
])

# ------------------------
# Prediction
# ------------------------
if st.button("Predict Disease"):

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
    breed_columns = [
        "breed_Labrador","breed_German Shepherd","breed_Bulldog",
        "breed_Poodle","breed_Beagle","breed_Persian Cat",
        "breed_Siamese Cat","breed_Golden Retriever",
        "breed_Rottweiler","breed_Maine Coon"
    ]

    for b in breed_columns:
        features[b] = 1 if b.split("_")[1] == breed else 0

    input_df = pd.DataFrame([features])
    input_df = input_df[model_columns]

    pred_class_num = disease_model.predict(input_df)[0]
    probability = np.max(disease_model.predict_proba(input_df))

    pred_class_label = le.inverse_transform([pred_class_num])[0]

    st.success(f"Predicted Condition: {pred_class_label}")
    st.info(f"Confidence: {probability*100:.2f}%")
    st.warning("⚠️ This is an AI-based prediction. Please consult a veterinarian.")