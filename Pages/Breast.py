import pandas as pd
from keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np
model=load_model("brease.keras")


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

# """
#  1   radius_mean             569 non-null    float64
#  2   texture_mean            569 non-null    float64
#  3   perimeter_mean          569 non-null    float64
#  4   smoothness_mean         569 non-null    float64
#  5   compactness_mean        569 non-null    float64
#  6   concavity_mean          569 non-null    float64
#  7   concave points_mean     569 non-null    float64
#  8   symmetry_mean           569 non-null    float64
#  9   fractal_dimension_mean  569 non-null    float64


# """

readius_mean=st.slider("Enter the radius mean",min_value=0,max_value=50,value=7)

texture_mean=st.slider("Emter the texture mean",min_value=0,max_value=60,value=21)

perimeter_mean=st.number_input("Enter the perimeter_mean")

smoothness_mean=st.number_input("Enter the smoothness_mean")

compactness_mean=st.number_input("Enter the compactness_mean")

concavity_mean=st.slider("Enter the concavity_mean",min_value=0.0,max_value=2.0,value=0.3)

concave_points_mean=st.slider("Enter the concave_points_mean",min_value=0.0,max_value=2.0,value=0.03)


symmetry_mean=st.slider("Enter the symmetry_mean",min_value=0.0,max_value=2.0,value=0.13)

fractal_dimension_mean=st.slider("Enter the fractal_dimension_mean",min_value=0.0,max_value=0.2,value=0.07)





def predict_data(radius_mean,texture_mean,perimeter_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean):
    
    input_values=[[radius_mean,texture_mean,perimeter_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean]]
    cols_name=["radius_mean","texture_mean","perimeter_mean","smoothness_mean","compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean"]

    pre_data = pd.DataFrame(
        input_values,
        columns=cols_name
    )
    return pre_data
    

predict_encoding={
    "B":0,
    "M":1
}

predict_encoding_inv={
    0:"B",
    1:"M"
}

if st.button("Predict"):
    
    result=np.round(model.predict(predict_data(radius_mean=readius_mean,texture_mean=texture_mean,perimeter_mean=perimeter_mean,smoothness_mean=smoothness_mean,compactness_mean=compactness_mean,concave_points_mean=concave_points_mean,symmetry_mean=symmetry_mean,fractal_dimension_mean=fractal_dimension_mean,concavity_mean=concavity_mean)))[0][0]
    
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
            ✅ Benign
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
            ❌ Malignant
        </div>
        """,
        unsafe_allow_html=True
    )
        st.write(result)
        
    st.subheader("📊 Model Performance")

    img1 = Image.open("accuracy_breast.png")
    img2 = Image.open("loss_breast.png")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img1, caption="Model Accuracy")

    with col2:
        st.image(img2, caption="Model Loss")
        
    
    
    
    