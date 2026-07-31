# app.py

import streamlit as st
import numpy as np
import joblib
from PIL import Image

from config import EDU_MAP, MARITAL_MAP, RESIDENCE_MAP, PURPOSE_MAP, FEATURE_COLUMNS

# Load and display logo
logo = Image.open("logo.png")
st.image(logo, width=100)

# Load model and scaler
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Risk Prediction System")
st.write("Enter the customer details below:")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Monthly Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_score = st.select_slider(
    "Credit Score",
    options=list(range(300, 901, 10)),
    value=600,
    format_func=lambda x: f"{x} points"
)
employment_years = st.number_input("Years of Employment", min_value=0, max_value=50)
education_level = st.selectbox("Education Level", list(EDU_MAP.keys()))
marital_status = st.selectbox("Marital Status", list(MARITAL_MAP.keys()))
num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10)
existing_loans_count = st.number_input("Existing Loans Count", min_value=0, max_value=10)
residence_type = st.selectbox("Residence Type", list(RESIDENCE_MAP.keys()))
loan_purpose = st.selectbox("Loan Purpose", list(PURPOSE_MAP.keys()))

# Build input vector in the exact column order used at training time
features = np.array([[
    age,
    income,
    loan_amount,
    credit_score,
    employment_years,
    EDU_MAP[education_level],
    MARITAL_MAP[marital_status],
    num_dependents,
    existing_loans_count,
    RESIDENCE_MAP[residence_type],
    PURPOSE_MAP[loan_purpose],
]])

assert features.shape[1] == len(FEATURE_COLUMNS), "Feature count mismatch with training config"

# Scale features
features_scaled = scaler.transform(features)

if st.button("Predict Loan Approval"):
    prediction = model.predict(features_scaled)[0]
    score = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Loan is likely to be Approved. Score: {score:.2f}")
    else:
        st.error(f"❌ Loan is likely to be Rejected. Score: {score:.2f}")

    with st.expander("Why this prediction? (top drivers)"):
        st.write(
            "This model's predictions are most influenced by **monthly income**, "
            "**existing loan count**, and **credit score**, in that order — "
            "see `feature_importance.png` in the repo for the full breakdown."
        )
