import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
stacking_clf = joblib.load("aqua_safe_model.pkl")

# --- PAGE STYLING ---
st.set_page_config(page_title="Aqua-Safe", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
        color: white;
    }

    .reportview-container .main .block-container {
        background-color: rgba(0, 0, 0, 0.6);  
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        color: white;  
    }

   
    h1 {
        color: #ffffff;
        font-family: 'Arial Black', sans-serif;
    }
    h2, h3 {
        color: #f0f0f0;
    }

    
    .stMarkdown p {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- TITLE AND DESCRIPTION ---
st.title("Aqua-Safe: Water Potability Classification")
st.markdown("""
Predict whether water is **Potable** or **Not Potable** based on water quality parameters.
Adjust the input values and threshold to see the prediction in real-time.
""")

# --- USER INPUT FUNCTION ---
def user_input_features():
    st.subheader("Enter Water Quality Parameters")
    ph = st.number_input("pH", 0.0, 14.0, 7.0, step=0.01)
    hardness = st.number_input("Hardness (mg/L)", 0.0, 1000.0, 200.0)
    solids = st.number_input("Solids (mg/L)", 0.0, 100000.0, 500.0)
    chloramines = st.number_input("Chloramines (mg/L)", 0.0, 20.0, 4.0)
    sulfate = st.number_input("Sulfate (mg/L)", 0.0, 500.0, 30.0)
    conductivity = st.number_input("Conductivity (µS/cm)", 0.0, 2000.0, 400.0)
    organic_carbon = st.number_input("Organic Carbon (mg/L)", 0.0, 50.0, 2.0)
    trihalomethanes = st.number_input("Trihalomethanes (µg/L)", 0.0, 200.0, 40.0)
    turbidity = st.number_input("Turbidity (NTU)", 0.0, 10.0, 1.0)
    
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
    df = pd.DataFrame([data])
    df["solids_per_conductivity"] = df["Solids"] / (df["Conductivity"] + 1)
    df["chloramine_sulfate_ratio"] = df["Chloramines"] / (df["Sulfate"] + 1)
    df["organic_turbidity"] = df["Organic_carbon"] * df["Turbidity"]
    df["ph_deviation"] = abs(df["ph"] - 7)

    return df

input_df = user_input_features()

# --- THRESHOLD SLIDER ---
st.subheader("Classification Threshold")
st.markdown("""
Adjust the threshold to control model sensitivity:
- **Higher threshold** → more conservative, fewer false positives (predict drinkable less often)
- **Lower threshold** → more likely to classify water as drinkable
""")
best_threshold = st.slider("Threshold", 0.0, 1.0, 0.54, 0.01)

# --- PREDICTION ---
y_proba = stacking_clf.predict_proba(input_df)[:, 1]
prediction = (y_proba >= best_threshold).astype(int)

st.subheader("Prediction")
if prediction[0] == 1:
    st.success(f"✅ Potable! Probability: {y_proba[0]*100:.2f}%")
else:
    st.error(f"⚠️ Not Potable! Probability: {y_proba[0]*100:.2f}%")

# --- OPTIONAL: Show input values ---
with st.expander("View Input Parameters"):
    st.dataframe(input_df)
