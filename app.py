import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🩺", layout="centered")
MODEL_PATH = Path(__file__).parent / "medical_insurance_linear_regression.joblib"
model = joblib.load(MODEL_PATH)

st.title("🩺 Medical Insurance Cost Predictor")
st.write("Enter customer details to estimate medical insurance charges.")
st.caption("Educational machine-learning demo. This estimate is not an insurance quote or medical/financial advice.")

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        age = st.slider("Age", min_value=18, max_value=100, value=30, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")
        sex = st.selectbox("Gender", options=["female", "male"], format_func=lambda x: x.title())
    with right:
        children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
        smoker = st.selectbox("Smoking status", options=["no", "yes"], format_func=lambda x: "Non-smoker" if x == "no" else "Smoker")
        region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"], format_func=lambda x: x.title())
    submitted = st.form_submit_button("Estimate insurance cost", use_container_width=True)

if submitted:
    new_customer = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": int(children),
        "sex": sex,
        "smoker": smoker,
        "region": region,
    }])
    estimate = float(model.predict(new_customer)[0])
    st.success(f"Estimated insurance cost: ${estimate:,.2f}")
    st.caption("The result is a model estimate; real premiums can differ.")
