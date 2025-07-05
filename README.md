# Vehicle-Maintenance-Predictor
An AI-based web application that predicts whether a vehicle requires maintenance, using real-world usage data and health indicators. Built with XGBoost and deployed using Streamlit, the system analyzes critical parameters like mileage, engine temperature, oil level, tire thread depth, brake wear, and service history to make accurate predictions.

Key Features:
  Predicts maintenance needs using a machine learning classification model,
   Uses MinMaxScaler to normalize vehicle parameter inputs,
   Built and trained using XGBoost with over 200 labeled samples,
   Achieves up to 97% accuracy on validation data,
   Web interface built with Streamlit for easy user input and real-time predictions,
   Debug mode shows raw/scaled inputs and model confidence,
   Deployed on Streamlit Cloud for instant access and testing.

Tech Stack:
  Python, Pandas, NumPy,
   XGBoost, Scikit-learn,
   Streamlit,
   Joblib (for model & scaler persistence).

How It Works:
  User enters vehicle usage and health parameters into the app,
   Input data is scaled using the same scaler used during training,
   The trained model predicts whether maintenance is required,
   The app displays a clear result along with prediction confidence.

**Code:**
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
st.title("Vehicle Maintenance Prediction System")
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
if st.button("Predict Maintenance Need"):
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
