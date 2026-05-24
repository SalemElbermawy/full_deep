# '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'
# housing_median_age	total_rooms	total_bedrooms	population	households	ocean_proximity
import pandas as pd
import streamlit as st
from keras.models import load_model
from PIL import Image

model=load_model("ca_housing.keras")


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

encoding={
    
    '<1H OCEAN':0,
    'INLAND':1,
    'ISLAND':2,
    'NEAR BAY':3,
    'NEAR OCEAN':4
    
}

def predict_house(housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity):
    
    inputs_value=[[housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity]]
    cols_names=["housing_median_age","total_rooms","total_bedrooms","population","households","median_income","ocean_proximity"]
    
    predict_data=pd.DataFrame(
        inputs_value,
        columns=cols_names
    )
    return predict_data

st.title("🏡 California Housing Price Predictor")

house_age=st.slider("Age of the house",min_value=1,max_value=60,value=27)

total_rooms=st.number_input("Enter Total Rooms",value=3,min_value=1)

total_bedrooms=st.number_input("Enter Total Bedrooms",value=3,min_value=1)

median_income=st.slider("Median Income",min_value=0,max_value=50,value=8)


households=st.slider("Households of the house",min_value=1,value=500,max_value=10000)

population=st.number_input("Enter The Number Of Population",min_value=2,value=3000)

ocean_proximity=encoding[st.selectbox("Choose :",options=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])]

if st.button("Predict"):
    result=model.predict(predict_house(housing_median_age=house_age,total_bedrooms=total_bedrooms,total_rooms=total_rooms,households=households,population=population,ocean_proximity=ocean_proximity,median_income=median_income))
    
    st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        color: white;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
        border: 3px solid #ffffff33;
        margin-top: 20px;
    ">
        <h3 style="margin-bottom:10px; font-size:28px;">
            🏡 Predicted House Price
        </h3>

        <h1 style="
            color:#FFD700;
            font-size:50px;
            margin:0;
        ">
            {result[0]}
        </h1>

        <p style="
            font-size:18px;
            opacity:0.9;
            margin-top:15px;
        ">
            AI Estimated Market Value
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
    
    st.subheader("📊 Model Performance")

    img1 = Image.open("mse_ca.png")
    img2 = Image.open("loss_ca.png")
    img3 = Image.open("mae_ca.png")
    img4=Image.open("r2_ca.png")

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

