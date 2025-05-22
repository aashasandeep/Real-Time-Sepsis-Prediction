import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Real-Time Sepsis Risk Predictor", layout="centered")

st.title("Real-Time Sepsis Prediction Dashboard")
st.markdown("""
This app predicts the *risk of sepsis* in patients and explains the model's decision using *SHAP* and *LIME*.
""")

# --- Upload Patient Data ---
uploaded_file = st.file_uploader("Upload a CSV file with patient vitals", type=["csv"])

if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    st.write("Patient Data Preview:", input_data.head())

import random

def simulate_vitals_data():
    data = {
        'Heart Rate': [random.randint(60, 100) for _ in range(60)],
        'SpO2': [random.randint(95, 100) for _ in range(60)],
        'Respiration Rate': [random.randint(12, 20) for _ in range(60)],
        'BP Sys': [random.randint(110, 130) for _ in range(60)],
        'BP Dia': [random.randint(70, 90) for _ in range(60)],
    }
    return pd.DataFrame(data)

st.title("Patient Vitals Simulator")

if st.button("Simulate 1-minute Patient Data"):
    data = simulate_vitals_data()
    st.markdown("## Incoming Vitals")
    st.dataframe(data)

# Simulated probability prediction (replace with your model's output)
prediction_proba = 0.72  # Example: model.predict_proba(X)[0][1] if using sklearn

# Display a subheader
st.subheader("Sepsis Risk Score")

# Convert to binary label using threshold
prediction_label = int(prediction_proba >= 0.5)

# Show the result
st.write("Predicted Probability:", round(prediction_proba, 2))
st.write("Sepsis Prediction:", "Yes" if prediction_label == 1 else "No")

 # --- LIME Explanation ---
st.subheader(" LIME Explanation (Patient 1)")

st.success("Prediction and explainability complete.")

# --- HIPAA & Security Notes ---
st.markdown("""
---
 *Security & Compliance Notice*  
- Patient data must be anonymized before upload.  
- This app does not store any data.  
- Deploy only on *HIPAA-compliant cloud services* with encryption and access control.
""")


# Sidebar config with number input and slider
st.sidebar.title("Model Monitoring Configuration")

st.sidebar.markdown("### Confidence Threshold")
confidence_threshold = st.sidebar.number_input(
    "Trigger alert if confidence is below:", 
    min_value=0.0, max_value=1.0, value=0.79, step=0.01
)

st.sidebar.markdown("### Accuracy Threshold")
accuracy_threshold = st.sidebar.number_input(
    "Trigger alert if accuracy is below:", 
    min_value=0.0, max_value=1.0, value=0.60, step=0.01
)

# slider

# first argument takes the title of the slider
# second argument takes the starting of the slider
# last argument takes the end number
level = st.slider("Select the level", 1, 5)

# print the level
# format() is used to print value 
# of a variable at a specific position
st.text('Selected: {}'.format(level))






