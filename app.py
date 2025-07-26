import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("xgb_model_2.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection App")
st.markdown("Enter transaction details below to check if it's fraudulent:")

# Input features (example: V1 to V4 + normalizedAmount)
V1 = st.number_input("V1", value=0.0)
V2 = st.number_input("V2", value=0.0)
V3 = st.number_input("V3", value=0.0)
V4 = st.number_input("V4", value=0.0)
normalizedAmount = st.number_input("Normalized Amount", value=0.0)

# Button to predict
if st.button("Predict Fraud"):
    input_data = pd.DataFrame([[V1, V2, V3, V4, normalizedAmount]],
                              columns=["V1", "V2", "V3", "V4", "normalizedAmount"])
    
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠ Fraudulent Transaction Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction. (Probability: {prob:.2f})")