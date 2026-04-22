import streamlit as st
import numpy as np
import joblib
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")
# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🩺 Diabetes Risk Prediction System")
st.markdown("### Early detection using Machine Learning")
st.write("---")

st.write("Enter patient details:")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 150)
    skin = st.number_input("Skin Thickness", 0, 100)

with col2:
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
    age = st.number_input("Age", 1, 120)

if st.button("🔍 Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.write("---")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Diabetes ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Diabetes ({(1-probability)*100:.2f}%)")

    st.progress(int(probability * 100))
st.write("---")
st.subheader("🧬 Why these factors matter")

st.markdown("""
- **Glucose**: High levels indicate poor insulin regulation  
- **BMI**: Higher BMI increases insulin resistance  
- **Age**: Risk increases with metabolic stress over time  
- **Insulin**: Indicates how well the body processes glucose  
""")
st.write("---")
st.markdown("Developed by CSE & Biotech Team | ML-Based Early Detection System")
