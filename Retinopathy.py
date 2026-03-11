import streamlit as st
import pickle
import pandas as pd

st.title("Diabetic Retinopathy Predictor")

# Load pre-trained model and scaler
model = pickle.load(open("logreg_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Input fields
st.write("### Enter Patient Data")
age = st.number_input("Age", min_value=10, max_value=90, value=50)
systolic_bp = st.number_input("Systolic BP", min_value=70, max_value=150, value=120)
diastolic_bp = st.number_input("Diastolic BP", min_value=60, max_value=120, value=80)
cholesterol = st.number_input("Cholesterol", min_value=70, max_value=130, value=100)

if st.button("Predict"):
    # Prepare input
    input_data = pd.DataFrame([[age, systolic_bp, diastolic_bp, cholesterol]],
                              columns=["age", "systolic_bp", "diastolic_bp", "cholesterol"])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    prob_percent = probability * 100
    
    # Show result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"Predicted: Diabetic Retinopathy ({prob_percent:.1f}%)")
    else:
        st.success(f"Predicted: No Retinopathy ({prob_percent:.1f}%)")
    
    # Show progress bar
    st.progress(int(prob_percent))
    
    # Optional: Show input data
    st.write("### Entered Patient Data")
    st.write(f"Age: {age}, Systolic BP: {systolic_bp}, Diastolic BP: {diastolic_bp}, Cholesterol: {cholesterol}")