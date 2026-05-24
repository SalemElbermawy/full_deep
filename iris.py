import pandas as pd
from keras.models import load_model
import streamlit as st
import numpy as np

from PIL import Image

map_encoding={
    
    0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'
    
}

colors = {
    'Iris-setosa': "#4CAF50",
    "Iris-versicolor": "#2196F3",
    "Iris-virginica": "#9C27B0"
}

model = load_model("iris.keras")



def prediction_function(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm):
    
    input_values=[[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]]
    cols_names=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
    
    predict_data=pd.DataFrame(input_values,columns=cols_names)
    
    return predict_data


st.markdown("""
<style>
div.stButton > button {
    background: linear-gradient(90deg, #4CAF50, #45a049);
    color: white;
    font-size: 22px;
    font-weight: bold;
    padding: 12px 40px;
    border-radius: 15px;
    border: none;
    width: 100%;
    transition: 0.3s;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}

div.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #45a049, #4CAF50);
}
</style>
""", unsafe_allow_html=True)



st.title("🌸 Iris Flower Prediction")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.3)
petal_width = st.slider("Petal Width", 0.1, 3.0, 1.3)


if st.button("Predict"):

    result=(map_encoding[(np.argmax(model.predict(prediction_function(SepalLengthCm=sepal_length,SepalWidthCm=sepal_width,PetalLengthCm=petal_length,PetalWidthCm=petal_width))))])
    
    color = colors[result]

    st.markdown(
        f"""
        <div style="
            background-color:{color};
            padding:25px;
            border-radius:20px;
            text-align:center;
            color:white;
            font-size:30px;
            font-weight:bold;
            box-shadow:0 4px 10px rgba(0,0,0,0.2);
        ">
            Prediction: {result}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("📊 Model Performance")

    img1 = Image.open("accuracy_iris.png")
    img2 = Image.open("loss_iris.png")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img1, caption="Model Accuracy")

    with col2:
        st.image(img2, caption="Model Loss")

    
    
    