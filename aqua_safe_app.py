import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

stacking_clf = joblib.load("aqua_safe_model.pkl")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #f0f8ff, #a2d2ff);
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Aqua-Safe: Water Potability Classification")
st.markdown("""
Predict whether water is **Potable** or **Not Potable** based on water quality parameters.
""")

def user_input_features():
    ph = st.number_input("pH", 0.0, 14.0, 7.0)
    hardness = st.number_input("Hardness", 0.0, 1000.0, 200.0)
    solids = st.number_input("Solids", 0.0, 100000.0, 500.0)
    chloramines = st.number_input("Chloramines", 0.0, 20.0, 4.0)
    sulfate = st.number_input("Sulfate", 0.0, 1000.0, 30.0)
    conductivity = st.number_input("Conductivity", 0.0, 2000.0, 400.0)
    organic_carbon = st.number_input("Organic Carbon", 0.0, 50.0, 2.0)
    trihalomethanes = st.number_input("Trihalomethanes", 0.0, 200.0, 40.0)
    turbidity = st.number_input("Turbidity", 0.0, 10.0, 1.0)
    
    data = {
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("Threshold for Classification")
best_threshold = st.slider("Threshold", 0.0, 1.0, 0.54, 0.01)

y_proba = stacking_clf.predict_proba(input_df)[:, 1]
prediction = (y_proba >= best_threshold).astype(int)

st.subheader("Prediction")
st.write("✅ Potable" if prediction[0] == 1 else "⚠️ Not Potable")
st.write(f"Probability of Potability: {y_proba[0]*100:.2f}%")




