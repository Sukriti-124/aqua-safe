import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

stacking_clf = joblib.load("aqua_safe_model.pkl")
st.title("Aqua-Safe: Water Potability Predictor")
st.markdown("""
Predict if the water is Potable or Not Potable based on water quality parameters
""")

