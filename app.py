import streamlit as st
import pickle
import pandas as pd

# Page setup
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    with open('california_knn_pipeline.pkl', 'rb') as f:
        model = joblib.load('california_knn_pipeline.joblib')
    return model

model = load_model()

# Title
st.title("🏠 California House Price Predictor")
st.write("Enter house details to predict price")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Economic Factors")
    med_inc = st.number_input("Median Income ($10k)", 0.0, 15.0, 3.0, 0.1)
    house_age = st.slider("House Age (years)", 1, 52, 25)
    ave_rooms = st.number_input("Average Rooms", 1.0, 20.0, 5.0, 0.1)
    ave_bedrms = st.number_input("Average Bedrooms", 0.5, 10.0, 2.0, 0.1)

with col2:
    st.subheader("📍 Location & Population")
    population = st.number_input("Population", 1, 10000, 1000, 10)
    ave_occup = st.number_input("Average Occupancy", 0.5, 15.0, 3.0, 0.1)
    latitude = st.number_input("Latitude", 32.0, 42.0, 34.0, 0.01)
    longitude = st.number_input("Longitude", -125.0, -114.0, -118.0, 0.01)

# Predict button
if st.button("🔮 Predict House Price", type="primary"):
    
    # Prepare input
    input_data = pd.DataFrame({
        'MedInc': [med_inc],
        'HouseAge': [house_age],
        'AveRooms': [ave_rooms],
        'AveBedrms': [ave_bedrms],
        'Population': [population],
        'AveOccup': [ave_occup],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    price = prediction * 100000
    
    # Display result
    st.success("✅ Prediction Complete!")
    st.metric(label="Estimated Price", value=f"${price:,.2f}")
