
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# Load model and scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('🏠 House Price Prediction App')
st.write('Enter the details below to predict the house price (in log scale):')

# User inputs
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
stories = st.slider("Stories", 1, 4, 2)
parking = st.slider("Parking spaces", 0, 4, 1)
log_area = st.slider("Log Area (sqft)", 6.0, 10.0, 8.5)
mainroad = st.selectbox("Main road access", ['No', 'Yes'])
guestroom = st.selectbox("Guestroom available", ['No', 'Yes'])
basement = st.selectbox("Basement available", ['No', 'Yes'])
hotwaterheating = st.selectbox("Hot Water Heating", ['No', 'Yes'])
airconditioning = st.selectbox("Air Conditioning", ['No', 'Yes'])
prefarea = st.selectbox("Pref Area", ['No', 'Yes'])
furnishingstatus = st.selectbox("Furnishing Status", ['Unfurnished', 'Semi-furnished', 'Furnished'])

# Prepare input
input_data = pd.DataFrame([[
    bedrooms, bathrooms, stories, parking, log_area,
    1 if mainroad == 'Yes' else 0,
    1 if guestroom == 'Yes' else 0,
    1 if basement == 'Yes' else 0,
    1 if hotwaterheating == 'Yes' else 0,
    1 if airconditioning == 'Yes' else 0,
    1 if prefarea == 'Yes' else 0,
    1 if furnishingstatus == 'Semi-furnished' else 0,
    1 if furnishingstatus == 'Furnished' else 0
]], columns=[
    'bedrooms', 'bathrooms', 'stories', 'parking', 'log_area',
    'mainroad_1', 'guestroom_1', 'basement_1', 'hotwaterheating_1',
    'airconditioning_1', 'prefarea_1', 'furnishingstatus_1', 'furnishingstatus_2'
])

# Predict
if st.button("Predict Price"):
    prediction_log = model.predict(input_data)[0]
    prediction = np.expm1(prediction_log)
    st.success(f"Predicted House Price: ₹{prediction:,.0f}")
