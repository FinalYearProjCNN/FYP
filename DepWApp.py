import streamlit as st
from cwarning import warning
from PIL import Image, ImageOps
from CNN import saved_model

st.title("Image Classification")
st.header("Facial Emotion Correlation with Mental Health Illness Classification ")
st.text("Upload an image for classification")

uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = saved_model(image, 'model.h5')
if label == 0:
    st.write("Patient requires a diagnosis in line with the warning")
else:
    st.write("Further investigation required to determine")
