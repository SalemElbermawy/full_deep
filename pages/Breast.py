import pandas as pd
from keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np

model = load_model("brease.keras")

st.title("Breast Cancer Prediction")

readius_mean = st.slider("Enter the radius mean", min_value=0, max_value=50, value=7)


texture_mean = st.slider("Enter the texture mean", min_value=0, max_value=60, value=21)

perimeter_mean = st.number_input("Enter the perimeter_mean")



smoothness_mean = st.number_input("Enter the smoothness_mean")
compactness_mean = st.number_input("Enter the compactness_mean")


concavity_mean = st.slider("Enter the concavity_mean", min_value=0.0, max_value=2.0, value=0.3)

concave_points_mean = st.slider("Enter the concave_points_mean", min_value=0.0, max_value=2.0, value=0.03)
symmetry_mean = st.slider("Enter the symmetry_mean", min_value=0.0, max_value=2.0, value=0.13)


fractal_dimension_mean = st.slider("Enter the fractal_dimension_mean", min_value=0.0, max_value=0.2, value=0.07)

def predict_data(radius_mean, texture_mean, perimeter_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean):
    input_values = [[radius_mean, texture_mean, perimeter_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean]]
    cols_name = ["radius_mean", "texture_mean", "perimeter_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean"]
    pre_data = pd.DataFrame(input_values, columns=cols_name)
    return pre_data

if st.button("Predict"):
    result = np.round(model.predict(predict_data(
        radius_mean=readius_mean,
        texture_mean=texture_mean,
        perimeter_mean=perimeter_mean,
        smoothness_mean=smoothness_mean,
        compactness_mean=compactness_mean,
        concave_points_mean=concave_points_mean,
        symmetry_mean=symmetry_mean,
        fractal_dimension_mean=fractal_dimension_mean,
        concavity_mean=concavity_mean
    )))[0][0]
    
    st.session_state.prediction_result = int(result)



if "prediction_result" in st.session_state:
    if st.session_state.prediction_result == 1:
        st.success("Benign")
    else:
        st.error("Malignant")

st.subheader("Model Performance")

img1 = Image.open("accuracy_breast.png")
img2 = Image.open("loss_breast.png")

col1, col2 = st.columns(2)
with col1:
    st.image(img1, caption="Model Accuracy")
with col2:
    st.image(img2, caption="Model Loss")