import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from datetime import datetime
from fpdf import FPDF

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="PetCare AI", page_icon="🐾")

st.title("🐾 PetCare AI Veterinary Assistant")
st.write("Enter your pet's health details to predict possible diseases.")

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
# PDF Generator (FPDF)
# ------------------------
def generate_pdf_report(
    age, weight, temperature, breed,
    vomiting, lethargy, appetite_loss,
    skin_lesions, breathing_difficulty, joint_pain,
    disease, confidence
):

    pdf = FPDF()
    pdf.add_page()

    # Logo
    try:
        pdf.image("petcare_logo.png", x=80, y=8, w=50)
        pdf.ln(30)
    except:
        pdf.ln(10)

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "PetCare AI Veterinary Assistant", ln=True, align="C")
    pdf.ln(5)

    # Date & Time
    pdf.set_font("Arial", "", 12)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Diagnosis Date & Time: {current_time}", ln=True)
    pdf.ln(10)

    # Table-like layout
    pdf.set_font("Arial", "B", 12)
    pdf.cell(95, 8, "Field", border=1)
    pdf.cell(95, 8, "Value", border=1, ln=True)

    pdf.set_font("Arial", "", 12)
    fields = [
        ("Age (years)", age),
        ("Weight (kg)", weight),
        ("Temperature (°C)", temperature),
        ("Breed", breed),
        ("Vomiting", vomiting),
        ("Lethargy", lethargy),
        ("Loss of Appetite", appetite_loss),
        ("Skin Lesions", skin_lesions),
        ("Breathing Difficulty", breathing_difficulty),
        ("Joint Pain", joint_pain),
        ("Predicted Disease", disease),
        ("Confidence Level", f"{confidence:.2f}%")
    ]

    for field, value in fields:
        pdf.cell(95, 8, str(field), border=1)
        pdf.cell(95, 8, str(value), border=1, ln=True)

    pdf.ln(10)

    # Disclaimer
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(
        0,
        6,
        "Disclaimer: This report is generated using an AI prediction model. "
        "Please consult a qualified veterinarian for accurate diagnosis and treatment."
    )

    pdf_output = pdf.output(dest="S").encode("latin-1")
    return io.BytesIO(pdf_output)

# ------------------------
# User Inputs
# ------------------------
st.subheader("Pet Health Information")

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

    st.success(f"Predicted Disease: {pred_class_label}")
    st.info(f"Confidence Level: {probability*100:.2f}%")

    st.warning("⚠ This is an AI-based prediction. Please consult a veterinarian.")

    # Generate PDF
    pdf = generate_pdf_report(
        age, weight, temperature, breed,
        vomiting, lethargy, appetite_loss,
        skin_lesions, breathing_difficulty, joint_pain,
        pred_class_label, probability*100
    )

    st.download_button(
        label="📄 Download Diagnosis Report",
        data=pdf,
        file_name="pet_diagnosis_report.pdf",
        mime="application/pdf"
    )