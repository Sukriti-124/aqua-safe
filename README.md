# Aqua-Safe: Water Potability Classification

An interactive machine learning application that predicts whether a water sample is **Drinkable (potable)** or **Not Drinkable** based on its physicochemical properties.

Live Demo: [Streamlit Web App](https://aqua-safe-acvjvpqxvjmgwfpndbwguq.streamlit.app/)  
*(Hosted with Streamlit, allowing real-time input and prediction)*

## 📌 Problem Statement

Access to clean drinking water is essential for health and well-being. This project builds a **binary classification model** to determine if water is safe to drink using measurable water quality features.

## 📊 Dataset

- Dataset Source: Kaggle Water Potability dataset - [Water_Potability_Dataset] (https://www.kaggle.com/datasets/adityakadiwal/water-potability)
-  The dataset contains water quality metrics for 3276 different water bodies.
- Target Variable: `Potability` (0 = Not Drinkable, 1 = Drinkable)  
- Input Features:
  - pH : Indicator of acidic or alkaline condition of water status. WHO has recommended maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52–6.83.
  - Hardness : Capacity of water to precipitate soap caused by Calcium and Magnesium in mg/L.
  - Solids : Total dissolved solids in ppm. Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.
  - Chloramines : Amount of Chloramines in ppm. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.
  - Sulfate : Amount of Sulfates dissolved in mg/L.
  - Conductivity : Electrical conductivity of water in μS/cm. ccording to WHO standards, EC value should not exceeded 400 μS/cm.
  - Organic Carbon : Amount of organic carbon in ppm.
  - Trihalomethanes : Amount of Trihalomethanes in μg/L. THM levels up to 80 ppm is considered safe in drinking water.
  - Turbidity : Measure of light emiting property of water in NTU.

## 🧠 Approach

1. Loaded and explored the dataset to understand feature distributions.  
2. Handled missing values and performed any necessary preprocessing.  
3. Conducted **Exploratory Data Analysis (EDA)** to investigate relationships between features and potability.  
4. Trained five machine learning models for binary classification.  
5. Evaluated models using metrics such as accuracy, precision, recall, and F1-score.  
6. Saved the best performing model (`aqua_safe_model.pkl`) for use in a Streamlit app.

## 🏆 Evaluation & Results
Best Model: Random Forest Classifier
- **Accuracy:** 67.22%
- **ROC-AUC:** 60.32%
- **Class 0 (Not Drinkable)**
  - Precision: 0.67
  - Recall: 0.92
  - F1-score: 0.77
- **Class 1 (Drinkable)**
  - Precision: 0.69
  - Recall: 0.29
  - F1-score: 0.41

## 🛠️ Project Files
- `aqua_safe.ipynb` – Main analysis and model training notebook
- `aqua_safe_app.py` – Streamlit application for prediction
- `aqua_safe_model.pkl` – Trained classification model
- `requirements.txt` – Python dependencies
- `README.md` – Project overview and instructions


## 🧰 Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn, Streamlit, Jupyter Notebook, Google Colab
