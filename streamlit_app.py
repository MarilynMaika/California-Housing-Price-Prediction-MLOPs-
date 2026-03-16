import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
with open("california_knn_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

st.title("California Housing Price Predictor")

st.write("Enter housing characteristics to estimate the median house value.")

# Input fields
MedInc = st.number_input("Median Income (MedInc)", min_value=0.0, step=0.1)
HouseAge = st.number_input("Median House Age", min_value=0.0, step=1.0)
AveRooms = st.number_input("Average Rooms per Household", min_value=0.0, step=0.1)
AveBedrms = st.number_input("Average Bedrooms per Household", min_value=0.0, step=0.1)
Population = st.number_input("Population", min_value=0.0, step=1.0)
AveOccup = st.number_input("Average Occupancy", min_value=0.0, step=0.1)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

# Prediction
if st.button("Predict House Value"):

    input_data = pd.DataFrame({
        "MedInc":[MedInc],
        "HouseAge":[HouseAge],
        "AveRooms":[AveRooms],
        "AveBedrms":[AveBedrms],
        "Population":[Population],
        "AveOccup":[AveOccup],
        "Latitude":[Latitude],
        "Longitude":[Longitude]
    })

    prediction = model.predict(input_data)

    price = prediction[0] * 100000

    st.success(f"Estimated Median House Value: ${price:,.2f}")