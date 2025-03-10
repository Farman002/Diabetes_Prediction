import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 🎯 Title and Description
st.title("Diabetes Prediction App")
st.write(
    "This app predicts if a person has diabetes based on the input features using a Gradient Boosting (GBDT) model.")

# 🔄 Load the Trained Model, Scaler, and Feature Columns
gbdt_model = joblib.load('gbdt_diabetes_model.joblib')
scaler = joblib.load('robust_scaler.joblib')
feature_columns = joblib.load('feature_columns.joblib')


# 🛠 Function to Get User Input
def get_user_input():
    # Create input fields for each feature
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.2f")
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    return user_data


# 🛠 Get user input
user_data = get_user_input()

# 🛠 Align input data with saved feature columns and fill missing with 0
user_data = user_data.reindex(columns=feature_columns).fillna(0)

# 🔄 Scale the input data
scaled_data = scaler.transform(user_data)

# 🛠 Make a prediction
prediction = gbdt_model.predict(scaled_data)[0]
prediction_proba = gbdt_model.predict_proba(scaled_data)[0]

# 🎯 Display Results
st.subheader("Prediction Result:")
if prediction == 1:
    st.error("The model predicts: **Diabetic**")
else:
    st.success("The model predicts: **Not Diabetic**")

st.subheader("Prediction Probability:")
st.write(f"Probability of being Diabetic: **{prediction_proba[1] * 100:.2f}%**")
st.write(f"Probability of being Not Diabetic: **{prediction_proba[0] * 100:.2f}%**")
