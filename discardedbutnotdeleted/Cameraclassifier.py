import import_ipynb
from tkinter import messagebox
import tkinter as tk
import matplotlib.pyplot as plt
import cv2
from cv2_plt_imshow import cv2_plt_imshow, plt_format
import tensorflow
import numpy as np
import keras
from keras.models import model_from_json
from keras.models import Sequential, save_model, load_model
from cwarning import show_warning

model = load_model('mymodel.h5')

model_json = model.to_json()
model.save_weights('model_weights.h5')
with open("model.json", "w") as json_file:
    json_file.write(model_json)


class FacialExpressionModel(object):
    EMOTIONS_LIST = ['angry', 'disgust', 'fear',
                     'happy', 'neutral', 'sad', 'surprise']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]


facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "mymodel.h5")
font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()
    # returns camera frames along with bounding boxes and predictions

    def get_frame(self):
        ret, fr = self.video.read()
        if not ret:
            return None
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return fr


def get_frame(self):
    ret, fr = self.video.read()
    if not ret:
        return None
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(fr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return fr


def gen(camera):
    while True:
        frame = camera.get_frame()
        cv2_plt_imshow('Facial Expression Recognization', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


gen(VideoCamera())
