import streamlit as st
import joblib
import numpy as np
st.set_page_config(page_title="My Iris Test")
st.title("Iris Flower Classification")
@st.cache_resource
def get_model():
    return joblib.load('iris_model.joblib')
st.image('myiris.png',caption='Iris Flower')
spl = st.text_input('Enter sepal length:','')
spw = st.text_input('Enter sepal width:','')
pel = st.text_input('Enter petal length:','')
pew = st.text_input('Enter petal width:','')
if st.button('Classify Flower'):
    values = [spl,spw,pel,pew]
    num_values = [float(x) for x in values]
    num_values_2D = np.asarray(num_values).reshape(1,-1) 
    knn_model = get_model()
    pred_value = knn_model.predict(num_values_2D)
    if pred_value == 0:
        st.write("This is setosa")
    elif pred_value == 1:    
        st.write("This is versicolor")
    else:
        st.write("This is virginica")