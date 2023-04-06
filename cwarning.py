import tkinter as tk
from tkinter import messagebox


# Classify the emotion
EMOTIONS_LIST = ['angry', 'disgust', 'fear',
                 'happy', 'neutral', 'sad', 'surprise']


# Define the elements and their corresponding diagnosis
element_diagnosis = {
    'angry': 'BPD',
    'disgust': 'OCD',
    'fear': 'Anxiety Disorder',
    'sad': 'Depression',
}

# Classify the emotion
emotion = [EMOTIONS_LIST]


def show_warning():
    if emotion == 'anger' and element_diagnosis['anger'] == 'BPD':
        warning("Higher risk for BPD!")
    if emotion == 'disgust' and element_diagnosis['disgust'] == 'OCD':
        warning("Higher risk for OCD!")
    if emotion == 'fear' and element_diagnosis['fear'] == 'Anxiety Disorder':
        warning("Higher risk for Anxiety Disorder!")
    if emotion == 'sad' and element_diagnosis['sad'] == 'Depression':
        warning("Higher risk for Depression!")


def warning(arg0):
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Warning", arg0)
