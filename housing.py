import pandas as pd
import streamlit as st
from keras.models import load_model
from PIL import Image

model = load_model("ca_housing.keras")

encoding = {
    '<1H OCEAN': 0,
    'INLAND': 1,
    'ISLAND': 2,
    'NEAR BAY': 3,
    'NEAR OCEAN': 4
}

def predict_house(housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity):
    inputs_value = [[housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity]]
    cols_names = ["housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"]
    
    predict_data = pd.DataFrame(inputs_value, columns=cols_names)
    return predict_data

st.title("California Housing Price Predictor")

house_age = st.slider("Age of the house", min_value=1, max_value=60, value=27)
total_rooms = st.number_input("Enter Total Rooms", value=3, min_value=1)
total_bedrooms = st.number_input("Enter Total Bedrooms", value=3, min_value=1)
median_income = st.slider("Median Income", min_value=0, max_value=50, value=8)
households = st.slider("Households of the house", min_value=1, value=500, max_value=10000)
population = st.number_input("Enter The Number Of Population", min_value=2, value=3000)

ocean_option = st.selectbox("Choose Ocean Proximity:", options=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])
ocean_proximity = encoding[ocean_option]

if st.button("Predict"):
    result = model.predict(predict_house(
        housing_median_age=house_age,
        total_bedrooms=total_bedrooms,
        total_rooms=total_rooms,
        households=households,
        population=population,
        ocean_proximity=ocean_proximity,
        median_income=median_income
    ))
    st.session_state.value = float(result[0][0])

if "value" in st.session_state:
    st.header("Predicted House Price")
    st.success(f"${st.session_state.value:,.2f}")

st.subheader("Model Performance")

img1 = Image.open("mse_ca.png")
img2 = Image.open("loss_ca.png")
img3 = Image.open("mae_ca.png")
img4 = Image.open("r2_ca.png")

col1, col2 = st.columns(2)
with col1:
    st.image(img1, caption="Model MSE")
with col2:
    st.image(img2, caption="Model Loss")
    
col3, col4 = st.columns(2)
with col3:
    st.image(img3, caption="Model MAE")
with col4:
    st.image(img4, caption="Model R2")