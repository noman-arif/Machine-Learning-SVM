import joblib
import streamlit as st
import numpy as np
import os

# Function to load the model
def load_model(filepath):
    if not os.path.exists(filepath):
        st.error("Error: Model file not found. Please ensure the file exists at the specified path.")
        st.stop()
    try:
        model = joblib.load(filepath)
        st.success(f"Model '{os.path.basename(filepath)}' loaded successfully!")
        return model
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        st.stop()

# Streamlit App
st.title("SVM Prediction Application")
st.write("Enter the required data for prediction:")

# Load the saved model and scaler
model_path = 'D:/UMT/ML/Python cide/F2021266586.sav'
scaler_path = 'D:/UMT/ML/Python cide/scaler.sav'
loaded_model = load_model(model_path)
loaded_scaler = load_model(scaler_path)

# Input fields
gender = st.radio("Gender", options=["Male", "Female"], help="Select the gender.")
age = st.number_input("Age", min_value=0, step=1, help="Enter age as an integer.")
estimated_salary = st.number_input("Estimated Salary", help="Enter the estimated salary.")

# Encode gender
gender_encoded = 1 if gender == "Male" else 0

if st.button("Predict"):
    # Prepare input data
    user_input = np.array([[gender_encoded, age, estimated_salary]])

    try:
        # Scale the input
        user_input_scaled = loaded_scaler.transform(user_input)

        # Predict
        prediction = loaded_model.predict(user_input_scaled)
        st.write("Prediction: Purchased" if prediction[0] == 1 else "Prediction: Not Purchased")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
