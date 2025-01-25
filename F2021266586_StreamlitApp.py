import streamlit as st
import numpy as np
import os
import pickle

# Function to load the model or scaler
def load_model(filepath):
    if not os.path.exists(filepath):
        st.error(f"Error: File not found at {filepath}. Please ensure the path is correct.")
        st.stop()
    try:
        # Attempt to load the file using pickle
        with open(filepath, 'rb') as file:
            obj = pickle.load(file)
        st.success(f"File '{os.path.basename(filepath)}' loaded successfully!")
        return obj
    except pickle.UnpicklingError:
        st.error(f"The file '{os.path.basename(filepath)}' is not a valid pickle file.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load the file '{os.path.basename(filepath)}'. Error: {e}")
        st.stop()

# Streamlit App
st.title("SVM Prediction Application")
st.write("Enter the required data for prediction:")

# Load the saved model and scaler
model_path = 'D:/UMT/Machine-Learning-SVM/F2021266586.pkl'
scaler_path = 'D:/UMT/Machine-Learning-SVM/scaler.pkl'
loaded_model = load_model(model_path)
loaded_scaler = load_model(scaler_path)

# Input fields
gender = st.radio("Gender", options=["Male", "Female"], help="Select the gender.")
age = st.number_input("Age", min_value=0, step=1, help="Enter age as an integer.")
estimated_salary = st.number_input("Estimated Salary", help="Enter the estimated salary.")

# Encode gender
gender_encoded = 1 if gender == "Male" else 0

# Prediction logic
if st.button("Predict"):
    try:
        # Prepare input data
        user_input = np.array([[gender_encoded, age, estimated_salary]])

        # Scale the input data
        user_input_scaled = loaded_scaler.transform(user_input)

        # Make the prediction
        prediction = loaded_model.predict(user_input_scaled)

        # Display the prediction result
        result = "Purchased" if prediction[0] == 1 else "Not Purchased"
        st.write(f"Prediction: **{result}**")
    except AttributeError as e:
        st.error(f"Error during prediction: {e}. Ensure the model and scaler are compatible.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
