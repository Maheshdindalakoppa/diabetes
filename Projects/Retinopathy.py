import streamlit as st
import pickle
import pandas as pd

# Load model and scaler
model = pickle.load(open("retinopathy_svm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Diabetic Retinopathy Prediction")
st.write("Enter patient data below:")

# Input fields for 4 features
age = st.number_input("Age", min_value=0, max_value=120, value=30)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=50, max_value=250, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=30, max_value=150, value=80)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)

# Run prediction only when button is clicked
if st.button("Predict"):
    # Create a DataFrame for input
    input_data = pd.DataFrame([[age, systolic_bp, diastolic_bp, cholesterol]],
                              columns=['age','systolic_bp','diastolic_bp','cholesterol'])
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Display results
    st.write(f"Predicted Class (0 = No Retinopathy, 1 = Retinopathy): {prediction}")
    st.write(f"Probability of Retinopathy: {probability:.2f}")