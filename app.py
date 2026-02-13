import streamlit as st
import joblib
import numpy as np
import re

# Load model and scaler
model = joblib.load("models/burnout_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Convert sleep text to numeric
def convert_sleep(value):
    numbers = re.findall(r'\d+', str(value))
    if len(numbers) == 2:
        return (int(numbers[0]) + int(numbers[1])) / 2
    elif len(numbers) == 1:
        return int(numbers[0])
    else:
        return 0

# App title
st.title("🎓 Student Burnout Detection System")
st.write("Enter student details to predict burnout level.")

# Input fields
sleep = st.selectbox("Average Sleep", ["2-4 hrs", "4-6 hrs", "7-8 hrs"])
academic_workload = st.slider("Academic Workload", 1, 5)
academic_pressure = st.slider("Academic Pressure", 1, 5)
financial_concerns = st.slider("Financial Concerns", 1, 5)
social_relationships = st.slider("Social Relationships", 1, 5)
depression = st.slider("Depression", 1, 5)
anxiety = st.slider("Anxiety", 1, 5)
isolation = st.slider("Isolation", 1, 5)
future_insecurity = st.slider("Future Insecurity", 1, 5)

if st.button("Predict Burnout"):

    sleep_numeric = convert_sleep(sleep)

    input_data = np.array([[
        sleep_numeric,
        academic_workload,
        academic_pressure,
        financial_concerns,
        social_relationships,
        depression,
        anxiety,
        isolation,
        future_insecurity
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.success("No Burnout ✅")
    elif prediction == 1:
        st.warning("Mild Burnout ⚠")
    else:
        st.error("Severe Burnout 🚨")

    st.write("Prediction Completed.")
