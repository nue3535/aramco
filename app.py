# Import necessary libraries
import streamlit as st
import numpy as np
import joblib

# Load the trained model from file
model = joblib.load("rf_model.joblib")

st.title("Diabetes Prediction App")

# Create input fields for all required features
pregnancies = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bloodpressure = st.number_input("BloodPressure")
skinthickness = st.number_input("SkinThickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
diabetespedigreefunction = st.number_input("DiabetesPedigreeFunction")
age = st.number_input("Age")

# Make prediction when the user clicks the button
if st.button("Predict"):
    # Combine all inputs into a single feature array
    features = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]])
    
    # Predict using the loaded model
    prediction = model.predict(features)

    # Display result
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    st.success(f"Prediction: {result}")
