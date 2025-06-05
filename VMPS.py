
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('xgboost_vehicle_model.pkl')

scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Vehicle Maintenance Predictor", layout="centered")
st.title("Vehicle Maintenance Prediction")
st.markdown("Enter recent vehicle data to predict whether service is required:")

mileage = st.number_input("Mileage (last 30 days in km)", value=2000)
enginetemp = st.number_input("Engine Temperature (°C)", value=85.0)
oillevel = st.slider("Oil Level (%)", 0, 100, 70)
tire = st.number_input("Tire Thread Depth (mm)", value=5.0)
brakes = st.number_input("Brake Life (mm)", value=4.0)
days_since_service = st.number_input("Days Since First Service", value=365)

if st.button("Predict Maintenance Need"):
    input_dict = {
        'Mileage (30 days)': mileage,
        'Engine Temperature (C):': enginetemp,
        'Oil Level (%)': oillevel,
        'Tire Thread (mm)': tire,
        'Brake Life (mm)': brakes,
        'Days Since First Service': days_since_service
    }
    input_df = pd.DataFrame([input_dict])

    expected_cols = ['Mileage (30 days)', 'Engine Temperature (C):', 'Oil Level (%)', 'Tire Thread (mm)', 'Brake Life (mm)', 'Days Since First Service']

    input_df = input_df[expected_cols]

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("Maintenance is Needed!")
    else:
        st.success("No Maintenance Needed")

    st.markdown("---")
    st.subheader("🛠 Debugging Info")
    st.write("Raw Input:", input_df)
    st.write("Scaled Input:", input_scaled)
    st.write("Prediction:", prediction)
    st.write("Prediction Probabilities:", prob)