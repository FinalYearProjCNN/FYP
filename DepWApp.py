import streamlit as st
from cwarning import show_warning
from PIL import Image, ImageOps
from keras.models import load_model
import numpy as np
from CNN import __init__

# Load the trained model
model = load_model('mymodel.h5')

# Define the class labels
class_labels = ['angry', 'disgusted', 'fearful',
                'happy', 'neutral', 'sad', 'surprised']

# Function to predict the class label of an image


def predict_class(image, model):
    # Preprocess the image
    image = ImageOps.fit(image, (48, 48))
    image = image.convert('RGB')
    image = np.array(image)
    image = image.reshape((1, 48, 48, 3))
    image = image / 255.0

    # Make prediction
    prediction = model.predict(image)
    return np.argmax(prediction)


st.title("Image Classification")
st.header("Facial Emotion Correlation with Mental Health Illness Classification ")
st.text("Upload an image for classification")

uploaded_file = st.file_uploader("Choose an image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict_class(image, model)
    if label == 0:
        show_warning.cwarning()
        st.write("Patient requires a diagnosis in line with the warning")
    else:
        st.write("Further investigation required to determine")
