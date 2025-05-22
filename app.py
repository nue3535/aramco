import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("rf_model.joblib")

st.title("Diabetes Prediction App")

# Assume your model takes these 3 features — adjust as needed

pregnancies = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bloodpressure = st.number_input("BloodPressure")
skinthickness = st.number_input("SkinThickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
diabetespedigreefunction = st.number_input("DiabetesPedigreeFunction")
age = st.number_input("Age")


if st.button("Predict"):
    features = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]])
    prediction = model.predict(features)
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.success(f"Prediction: {result}")
