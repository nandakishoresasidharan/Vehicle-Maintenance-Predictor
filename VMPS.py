# Vehicle maintenance prediction system

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load('xgboost_vehicle_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page setup
st.set_page_config(page_title="Vehicle Maintenance Predictor", layout="centered")
st.title("🚗 Vehicle Maintenance Prediction System")
st.markdown("Enter recent vehicle data to predict whether service is required:")

# User inputs
mileage = st.number_input("Mileage (last 30 days in km)", value=2000)
enginetemp = st.number_input("Engine Temperature (°C)", value=85.0)
oillevel = st.slider("Oil Level (%)", 0, 100, 70)
tire = st.number_input("Tire Thread Depth (mm)", value=5.0)
brakes = st.number_input("Brake Life (mm)", value=4.0)
days_since_service = st.number_input("Days Since First Service", value=365)

# Create input DataFrame
input_data = pd.DataFrame([[mileage, enginetemp, oillevel, tire, brakes, days_since_service]],
    columns=['Mileage (30 days)', 'Engine Temperature (C):', 'Oil Level (%)', 'Tire Thread (mm)',
             'Brake Life (mm)', 'Days Since First Service'])

# Scale the input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("🔍 Predict Maintenance Need"):
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error(f"Maintenance is Needed! (Confidence: {prob[0][1]*100:.2f}%)")
    else:
        st.success(f"No Maintenance Needed (Confidence: {prob[0][0]*100:.2f}%)")

    # Debug info
    st.markdown("---")
    st.subheader("Debugging Info")
    st.write("Raw Input:", input_data)
    st.write("Scaled Input:", input_scaled)
    st.write("Prediction Probabilities:", prob)
