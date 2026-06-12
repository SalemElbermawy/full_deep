import pandas as pd
from keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

model = load_model("iris.keras")

map_encoding = {
    0: 'Iris-setosa', 
    1: 'Iris-versicolor', 
    2: 'Iris-virginica'
}

def prediction_function(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):
    
    
    
    input_values = [[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]]
    
    cols_names = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    predict_data = pd.DataFrame(input_values, columns=cols_names)
    
    return predict_data

st.title("Iris Flower Prediction")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8)

sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.0)

petal_length = st.slider("Petal Length", 1.0, 7.0, 4.3)


petal_width = st.slider("Petal Width", 0.1, 3.0, 1.3)



if st.button("Predict"):
    
    result = map_encoding[np.argmax(model.predict(prediction_function(
        SepalLengthCm=sepal_length, 
        SepalWidthCm=sepal_width, 
        PetalLengthCm=petal_length, 
        PetalWidthCm=petal_width
    )))]
    
    st.session_state.prediction_result = result




if "prediction_result" in st.session_state:
    st.header("Prediction Result")
    st.success(st.session_state.prediction_result)



st.subheader("Model Performance")

img1 = Image.open("accuracy_iris.png")
img2 = Image.open("loss_iris.png")

col1, col2 = st.columns(2)

with col1:
    st.image(img1, caption="Model Accuracy")
    
    
with col2:
    st.image(img2, caption="Model Loss")