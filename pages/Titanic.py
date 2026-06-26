import pandas as pd
from keras.models import load_model
import streamlit as st
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


embarked_value= encoding_embarked[st.selectbox("Choose Embarked Value",options=["S","C","Q"],key="embarked")]

sex_value= sex_encoding[st.selectbox("Choose The Gender",options=["Male","Female"],key="sex")]
pclass=st.selectbox("Enter The pclass",options=[1,2,3],key="pclass")

age=st.number_input("Enter The Age")
fare=st.number_input("Enter the fare")
parch=st.number_input("Enter The parch")
sibsp=st.number_input("Enter The sibsp")

def predict(sex,age,fare,embarked,pclass,parch,sibsp):
    inputs_value=[[pclass,sex,age,sibsp,parch,fare,embarked]]
    col_names=["pclass","sex","age","sibsp","parch","fare","embarked"]
    df_predict=pd.DataFrame(inputs_value,columns=col_names)
    return df_predict


if st.button("Predict"):
    
    result=np.round(model.predict(predict(embarked_value,sex_value,pclass,age,parch,sibsp,fare)))[0][0]
    st.session_state.prediction_result=int(result)
    
if "prediction_result" in st.session_state:
    if st.session_state.prediction_result ==1:
        st.success("Survived")
    else:
        st.error("Did Not Survive")
        
st.subheader("Model Performance")

img1 = Image.open("accuracy_titanic.png")
img2=Image.open("loss_titanic.png")

col1,col2=st.columns(2)

with col1:
    
    st.image(img1, caption="Model Accuracy")
with col2:
    st.image(img2, caption="Model Loss")
