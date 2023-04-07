from flask import Flask, render_template, request, jsonify
import cv2
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('mymodel.h5')

# Define the class labels
class_labels = ['angry', 'disgusted', 'fearful',
                'happy', 'neutral', 'sad', 'surprised']

# Function to preprocess video frames


def preprocess_frame(frame):
    # Resize the frame to match the input requirements of your model
    frame = cv2.resize(frame, (48, 48))
    # Normalize the pixel values to [0, 1]
    frame = frame / 255.0
    # Convert the frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Reshape the frame to match the input shape of your model
    frame = np.array(frame)
    frame = frame.reshape((1, 48, 48, 3))
    return frame

# API endpoint to receive video frames and perform inference


@app.route('/campredict', methods=['POST'])
def predict():
    # Get the video frame from the request
    frame = request.files['frame'].read()
    nparr = np.fromstring(frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess the video frame
    processed_frame = preprocess_frame(frame)

    # Make prediction
    prediction = model.predict(processed_frame)
    label_idx = np.argmax(prediction)
    label = class_labels[label_idx]
    confidence = prediction[0][label_idx]

    # Return the predicted label and confidence as JSON response
    response = {
        'label': label,
        'confidence': confidence
    }
    return jsonify(response)

# Route to render the web app page


@app.route('/')
def index():
    return render_template('campredict.html')


if __name__ == '__main__':
    app.run(debug=True)
