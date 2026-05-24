from keras.models import load_model
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

model=load_model("titanic_model.keras")

encoding_embarked={
    "S":1,
    "C":2,
    "Q":3
}

sex_encoding={
    "Male":1,
    "Female":0
}

st.title("Titanic Detection")


embarked_value=encoding_embarked[st.selectbox("Choose The Embarked",options=["S","C","Q"],key="embarked")]
sex_value=sex_encoding[st.selectbox("Choose The Gender",options=["Male","Female"],key="sex")]
pclass = st.selectbox("Enter The pclass",options=[1,2,3],key="pclass")

age = st.number_input("Enter The age")
fare = st.number_input("Enter The fare")
parch = st.number_input("Enter The parch")
sibsp = st.number_input("Enter The sibsp")


def predict(sex,age,fare,embarked,pclass,parch,sibsp):
    
    inputs_value=[[pclass,sex,age,sibsp,parch,fare,embarked]]
    col_names=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
    
    df_predict=pd.DataFrame(
        inputs_value,
        columns=col_names
    )
    
    return df_predict


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


if st.button("Predict"):
    result=(np.round(model.predict(predict(embarked=embarked_value,sex=sex_value,pclass=pclass,age=age,parch=parch,sibsp=sibsp,fare=fare)))[0][0])

    if result == 1:
        st.markdown(
        """
        <div style="
            background-color:#d4edda;
            padding:20px;
            border-radius:10px;
            color:#155724;
            font-size:24px;
            text-align:center;
        ">
            ✅ Survived
        </div>
        """,
        unsafe_allow_html=True
    )
        
    else:
        st.markdown(
        """
        <div style="
            background-color:#f8d7da;
            padding:20px;
            border-radius:10px;
            color:#721c24;
            font-size:24px;
            text-align:center;
        ">
            ❌ Did Not Survive
        </div>
        """,
        unsafe_allow_html=True
    )
        
    st.subheader("📊 Model Performance")

    img1 = Image.open("accuracy_titanic.png")
    img2 = Image.open("loss_titanic.png")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img1, caption="Model Accuracy")

    with col2:
        st.image(img2, caption="Model Loss")

    
    
    
    
    