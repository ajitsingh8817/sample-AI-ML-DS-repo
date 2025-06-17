import streamlit as st
import pickle
import numpy as np
import os

st.title("🏦 Loan Approval Prediction App")

# Show current directory and files (debugging)
st.write("📂 Current Directory:", os.getcwd())
st.write("📄 Files Present:", os.listdir())

# Load model
model = None
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error(f"❌ Error loading model.pkl: {e}")

if model:
    # Input form
    st.header("📝 Applicant Details")

    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    LoanAmount = st.slider("Loan Amount (in thousands)", 0, 700, step=10)

    if st.button("🔮 Predict Loan Approval"):
        gender = 1 if Gender == "Male" else 0
        married = 1 if Married == "Yes" else 0
        credit = float(Credit_History)

        input_data = np.array([[gender, married, ApplicantIncome, LoanAmount, credit]])

        try:
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.success("✅ Loan Approved!")
            else:
                st.error("❌ Loan Not Approved.")
        except Exception as e:
            st.error(f"Prediction Failed: {e}")
else:
    st.warning("⚠️ Model not loaded. Prediction unavailable.")
