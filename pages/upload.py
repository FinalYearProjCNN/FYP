import streamlit as st
from cwarning import show_warning
from PIL import Image, ImageOps
from keras.models import load_model
import numpy as np
from discardedbutnotdeleted.CNN import __init__
from flask import Flask, render_template, request
import os

app = Flask(__name__)

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


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method != 'POST':
        return render_template('upload.html')
    # Get the uploaded file from the request
    upfile = request.files['file']

    if not upfile or not allowed_file(upfile.filename):
        return "Invalid file. Please upload a JPG image."
    # Save the uploaded file to a temporary location
    file_path = os.path.join('uploads', upfile.filename)
    upfile.save(file_path)

    # Open the uploaded file as an image
    image = Image.open(file_path)

    # Call the function to predict the class label
    label = predict_class(image, model)

    if label != 0:
        return "Further investigation required to determine"
    show_warning()
    return "Patient requires a diagnosis in line with the warning"


def allowed_file(filename):
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


if __name__ == '__main__':
    app.run(debug=True)
