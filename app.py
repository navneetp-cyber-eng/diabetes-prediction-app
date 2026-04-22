import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Diabetes Risk Analyzer", page_icon="🩺", layout="wide")

# Load model
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Custom styling
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #0b3c5d;
}
.sub-title {
    font-size: 18px;
    color: #555;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="big-title">🩺 Diabetes Risk Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-powered early detection system for diabetes risk assessment</p>', unsafe_allow_html=True)
st.write("---")

# Layout
col1, col2 = st.columns([1, 1])

# Input Section
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Patient Information")

    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level", 0, 200, 100)
    bp = st.slider("Blood Pressure", 0, 150, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("Age", 1, 120, 30)

    predict_button = st.button("🔍 Analyze Risk")
    st.markdown('</div>', unsafe_allow_html=True)

# Output Section
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if predict_button:
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction[0] == 1:
            st.error(f"⚠️ High Risk of Diabetes\n\nRisk Score: {probability*100:.2f}%")
        else:
            st.success(f"✅ Low Risk of Diabetes\n\nConfidence: {(1-probability)*100:.2f}%")

        st.progress(int(probability * 100))

    else:
        st.info("Enter patient details and click **Analyze Risk**")

    st.markdown('</div>', unsafe_allow_html=True)

# Info Section
st.write("---")
st.subheader("🧬 Clinical Insights")

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("**Glucose**\n\nIndicates blood sugar levels and insulin response.")

with col4:
    st.markdown("**BMI**\n\nHigher BMI is linked to insulin resistance.")

with col5:
    st.markdown("**Age**\n\nRisk increases with long-term metabolic stress.")

# Footer
st.write("---")
st.markdown("🔬 Developed for Early Detection of Diabetes | CSE + Biotech Project")
