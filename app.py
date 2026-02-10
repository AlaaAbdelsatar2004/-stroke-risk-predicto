import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('stroke_model.pkl')

st.title("Stroke Risk Predictor")
st.caption("Educational tool – Not a substitute for medical advice or doctor visit")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 0, 100, 35)
hypertension = st.radio("Do you have high blood pressure?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
heart_disease = st.radio("Do you have any heart disease?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
ever_married = st.selectbox("Have you ever been married?", ["Yes", "No"])
work_type = st.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence = st.selectbox("Residence type", ["Urban", "Rural"])
avg_glucose = st.number_input("Average glucose level (mg/dL)", 50.0, 300.0, 100.0, step=1.0)
bmi = st.number_input("Body Mass Index (BMI)", 10.0, 70.0, 25.0, step=0.5)
smoking = st.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

if st.button("Calculate Risk"):
    data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence],
        'avg_glucose_level': [avg_glucose],
        'bmi': [bmi],
        'smoking_status': [smoking]
    })

    prob = model.predict_proba(data)[0][1] * 100

    if prob > 20:
        st.error(f"Relatively **high** risk: {prob:.1f}%")
        st.warning("→ We strongly recommend consulting a doctor as soon as possible for a proper evaluation.")
    elif prob > 5:
        st.warning(f"**Moderate** risk: {prob:.1f}%")
        st.info("Keep regular follow-up with your doctor and maintain a healthy lifestyle.")
    else:
        st.success(f"**Low** risk: {prob:.1f}%")
        st.info("Continue with a healthy lifestyle to keep the risk low.")