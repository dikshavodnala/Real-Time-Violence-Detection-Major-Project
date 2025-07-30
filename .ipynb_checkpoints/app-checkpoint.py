import tensorflow as tf
from tensorflow import keras
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, Response
from tensorflow.keras.models import load_model


# Create flask app
app = Flask(__name__)
model = load_model("lrcn_model.h5")


# Define class names (replace with your actual class names)

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["crime", "non-crime"]

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    video_path = request.form['videoInput']
    prediction_result = predict_violence(video_path)
    return Response(prediction_result)


# Prediction function
def predict_violence(video_file_path):
    video_reader = cv2.VideoCapture(video_file_path)
    if not video_reader.isOpened():
        return "Error: Could not open video."

    frames_list = []
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)

    video_reader.release()

    if len(frames_list) < SEQUENCE_LENGTH:
        return "Error: Not enough frames extracted from video."

    predicted_probs = model.predict(np.expand_dims(frames_list, axis=0))[0]
    predicted_label = np.argmax(predicted_probs)
    predicted_class_name = CLASSES_LIST[predicted_label]
    confidence = predicted_probs[predicted_label]

    return f'Predicted: {predicted_class_name}\nConfidence: {confidence:.2f}'

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame,(IMAGE_HEIGHT,IMAGE_WIDTH))
    return resized_frame/255.0


def predict_violence_score(frame):
    # Predict violence score for a single frame
    return model.predict(np.expand_dims(frame, axis=0))[0][1]


if __name__ == "__main__":
    app.run(debug=True)