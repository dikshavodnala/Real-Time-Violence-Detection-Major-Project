import tensorflow as tf
from tensorflow import keras
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from pytz import timezone

# app = Flask(__name__)
# # Load the trained model
# model = keras.models.load_model("lrcn_model.h5")
# # model = keras.models.load_model("cnn_i3d_lstm_model (1).h5")
#
# # Define constants
# IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
# SEQUENCE_LENGTH = 20
# # CLASSES_LIST = ["non-violence", "violence"]
#
# #for lrcn model
# CLASSES_LIST = ["violence", "non-violence"]
#
#
# # Ensure "uploads" directory exists
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#
# @app.route('/')
# def home():
#     return render_template("index.html", prediction_text="Prediction will appear here...")
#
#
# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'videoInput' not in request.files:
#         print("Error: No file found in request")
#         return jsonify({"error": "No file selected."}), 400
#
#     video_file = request.files['videoInput']
#     if video_file.filename == '':
#         print("Error: No file chosen")
#         return jsonify({"error": "No file chosen."}), 400
#
#     # Save the uploaded file
#     video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
#     video_path = os.path.abspath(video_path).replace("\\", "/")
#
#     try:
#         video_file.save(video_path)
#         print(f"Video saved at: {video_path}")
#
#         prediction_text = predict_violence(video_path)
#         # print(f"Prediction Result: {prediction_class} ({confidence_score:.2f})")
#
#         return jsonify({"prediction": prediction_text})
#     except Exception as e:
#         print(f"Error occurred: {str(e)}")
#         return jsonify({"error": "An error occurred. Please try again."}), 500
#
# def send_email_notification(prediction_text):
#     sender_email = "majorproject694@gmail.com"
#     sender_password = "qshjjhtnitifvoia"
#     receiver_email = "vodnaladiksha123@gmail.com"
#
#     subject = "Violence Detection Alert!"
#     body = f"Alert! Suspicious activity detected.\n\nDetails:\n{prediction_text}"
#
#     msg = MIMEMultipart()
#     msg["From"] = sender_email
#     msg["To"] = receiver_email
#     msg["Subject"] = subject
#     msg.attach(MIMEText(body, "plain"))
#
#     try:
#         print("Connecting to SMTP server...")
#         server = smtplib.SMTP("smtp.gmail.com", 587)
#         server.ehlo()
#         server.starttls()
#         server.ehlo()
#         print("Logging in...")
#         server.login(sender_email, sender_password)
#         print("Sending email...")
#         server.sendmail(sender_email, receiver_email, msg.as_string())
#         server.quit()
#         print("Email notification sent successfully.")
#     except smtplib.SMTPAuthenticationError:
#         print("Authentication error: Check your app password.")
#     except smtplib.SMTPException as e:
#         print(f"SMTP error: {e}")
#     except Exception as e:
#         print(f"Error sending email: {e}")
#
#
# def predict_violence(video_file_path):
#     notification_message = None
#     print(f"Processing video: {video_file_path}")
#     os.chmod(video_file_path, 0o644)
#     video_reader = cv2.VideoCapture(video_file_path, cv2.CAP_FFMPEG)
#
#     if not video_reader.isOpened():
#         print(f"Error: Could not open video file..................")
#         return "Error: Could not open video file................."
#
#     # Check frame count
#     frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
#     if frame_count == 0:
#         print(f"Error: Video has zero frames.")
#         return "Error: Video has zero frames."
#     if not video_reader.isOpened():
#         print("Error: Could not open video.")
#         return "Error: Could not open video."
#
#     frames_list = []
#     video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"Total frames in video: {video_frames_count}")
#
#     if video_frames_count == 0:
#         print("Error: Video has no frames.")
#         return "Error: Video has no frames."
#
#     skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
#     print(f"Skipping {skip_frames_window} frames per selection")
#
#     for frame_counter in range(SEQUENCE_LENGTH):
#         video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
#         success, frame = video_reader.read()
#
#         if not success:
#             print(f"Error: Could not read frame at {frame_counter * skip_frames_window}")
#             return "Error: Could not read frames from video."
#
#         resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
#         normalized_frame = resized_frame / 255.0
#         frames_list.append(normalized_frame)
#
#     video_reader.release()
#
#     if len(frames_list) < SEQUENCE_LENGTH:
#         print("Error: Not enough frames extracted from video.")
#         return "Error: Not enough frames extracted from video."
#
#     predicted_probs = model.predict(np.expand_dims(frames_list, axis=0))[0]
#     predicted_label = np.argmax(predicted_probs)
#     predicted_class_name = CLASSES_LIST[predicted_label]
#     confidence = float(predicted_probs[predicted_label])
#     print(f"Predicted Probabilities: {predicted_probs}")
#     print(f"Predicted Class: {predicted_class_name}")
#
#
#     prediction_text = f"Prediction: {predicted_class_name} (Confidence: {confidence:.2f})"
#     print(prediction_text)
#     print(predicted_class_name)
#
#     # Send email if crime is detected
#     if predicted_class_name == "violence":
#         send_email_notification(prediction_text)
#
#     return prediction_text
#
#
# if __name__ == "__main__":
#     app.run(debug=True)



#real-time without email
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time

# Load the trained model
MODEL_PATH = "lrcn_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20

# Class labels
CLASSES_LIST = ["violence", "non-violence"]

# Initialize a queue for storing 20 frames
frame_queue = deque(maxlen=SEQUENCE_LENGTH)

# Function to preprocess frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))  # Resize
    frame = frame.astype("float32") / 255.0  # Normalize
    return frame

# Open webcam for real-time detection
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Preprocess and add frame to queue
    processed_frame = preprocess_frame(frame)
    frame_queue.append(processed_frame)

    # Predict only if we have 20 frames
    if len(frame_queue) == SEQUENCE_LENGTH:
        input_sequence = np.expand_dims(np.array(frame_queue), axis=0)  # Shape: (1, 20, 64, 64, 3)
        predicted_probs = model.predict(input_sequence)[0]  # Get probabilities for both classes

        # Identify class with highest probability
        predicted_label = np.argmax(predicted_probs)
        # predicted_class_name = CLASSES_LIST[predicted_label]
        confidence = float(predicted_probs[predicted_label])

        if predicted_probs[0] > 0.5:  # Adjust threshold
            predicted_class_name = "violence"
        else:
            predicted_class_name = "non-violence"

        print(f"Predicted Probabilities: {predicted_probs}")
        print(f"Predicted Class: {predicted_class_name}")

        # Display results on video feed
        label_text = f"{predicted_class_name}"
        color = (0, 0, 255) if predicted_class_name == "violence" else (0, 255, 0)

        cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        prediction_text = f"Prediction: {predicted_class_name} (Confidence: {confidence:.2f})"


    # Show video feed
    cv2.imshow("Real-Time Violence Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import tensorflow as tf
# from collections import deque
# import time
#
# # Load the trained model
# MODEL_PATH = "lrcn_model.h5"
# model = tf.keras.models.load_model(MODEL_PATH)
#
# # Define constants
# IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
# SEQUENCE_LENGTH = 20
# EMAIL_INTERVAL = 30  # Time interval (in seconds) between email notifications
#
# # Initialize a variable to store the last email sent time
# last_email_time = 0  # Stores the timestamp of the last sent email
#
# # Class labels
# CLASSES_LIST = ["violence", "non-violence"]
#
# # Initialize a queue for storing 20 frames
# frame_queue = deque(maxlen=SEQUENCE_LENGTH)
#
# # Function to preprocess frames
# def preprocess_frame(frame):
#     frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))  # Resize
#     frame = frame.astype("float32") / 255.0  # Normalize
#     return frame
#
#
# def send_email_notification(prediction_text):
#     global last_email_time
#     current_time = time.time()  # Get current timestamp
#
#     if current_time - last_email_time < EMAIL_INTERVAL:
#         print("Skipping email: Time interval not reached.")
#         return  # Exit the function without sending an email
#
#     sender_email = "majorproject694@gmail.com"
#     sender_password = "qshjjhtnitifvoia"
#     receiver_email = "vodnaladiksha123@gmail.com"
#
#     subject = "Violence Detection Alert!"
#     body = f"Alert! Suspicious activity detected.\n\nDetails:\n{prediction_text}"
#
#     msg = MIMEMultipart()
#     msg["From"] = sender_email
#     msg["To"] = receiver_email
#     msg["Subject"] = subject
#     msg.attach(MIMEText(body, "plain"))
#
#     try:
#         print("Connecting to SMTP server...")
#         server = smtplib.SMTP("smtp.gmail.com", 587)
#         server.ehlo()
#         server.starttls()
#         server.ehlo()
#         print("Logging in...")
#         server.login(sender_email, sender_password)
#         print("Sending email...")
#         server.sendmail(sender_email, receiver_email, msg.as_string())
#         server.quit()
#         print("Email notification sent successfully.")
#
#         last_email_time = current_time  # Update last email timestamp
#     except smtplib.SMTPAuthenticationError:
#         print("Authentication error: Check your app password.")
#     except smtplib.SMTPException as e:
#         print(f"SMTP error: {e}")
#     except Exception as e:
#         print(f"Error sending email: {e}")
#
# # Open webcam for real-time detection
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame.")
#         break
#
#     # Preprocess and add frame to queue
#     processed_frame = preprocess_frame(frame)
#     frame_queue.append(processed_frame)
#
#     # Predict only if we have 20 frames
#     if len(frame_queue) == SEQUENCE_LENGTH:
#         input_sequence = np.expand_dims(np.array(frame_queue), axis=0)  # Shape: (1, 20, 64, 64, 3)
#         predicted_probs = model.predict(input_sequence)[0]  # Get probabilities for both classes
#
#         # Identify class with highest probability
#         predicted_label = np.argmax(predicted_probs)
#         # predicted_class_name = CLASSES_LIST[predicted_label]
#         confidence = float(predicted_probs[predicted_label])
#
#         if predicted_probs[0] > 0.5:  # Adjust threshold
#             predicted_class_name = "violence"
#         else:
#             predicted_class_name = "non-violence"
#
#         print(f"Predicted Probabilities: {predicted_probs}")
#         print(f"Predicted Class: {predicted_class_name}")
#
#         # Display results on video feed
#         label_text = f"{predicted_class_name}"
#         color = (0, 0, 255) if predicted_class_name == "violence" else (0, 255, 0)
#
#         cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#         prediction_text = f"Prediction: {predicted_class_name} (Confidence: {confidence:.2f})"
#
#         # Alert if violence is detected
#         if predicted_class_name == "violence":
#             send_email_notification(prediction_text)
#
#
#     # Show video feed
#     cv2.imshow("Real-Time Violence Detection", frame)
#
#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


#
# real time crime detection code  using Camo Studio
# LRCN MODEL
# import cv2
# import numpy as np
# import tensorflow as tf
# from collections import deque
#
# # Load the trained model
# MODEL_PATH = "lrcn_model.h5"
# model = tf.keras.models.load_model(MODEL_PATH)
#
# # Define constants
# IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
# SEQUENCE_LENGTH = 20
#
# # Class labels
# CLASSES_LIST = ["crime", "non-crime"]
#
# # Initialize a queue for storing 20 frames
# frame_queue = deque(maxlen=SEQUENCE_LENGTH)
#
# # Function to preprocess frames
# def preprocess_frame(frame):
#     frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))  # Resize
#     frame = frame.astype("float32") / 255.0  # Normalize
#     return frame
#
# # Open Camo Studio virtual webcam
# CAMERA_INDEX = 1 # Change this if needed (try 0, 1, 2, etc., if not working)
# cap = cv2.VideoCapture(CAMERA_INDEX)
#
# if not cap.isOpened():
#     print("Error: Could not open Camo Studio webcam.")
#     exit()
#
# if not cap.isOpened():
#     print("Error: Could not open Camo Studio webcam.")
#     exit()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame.")
#         break
#
#     # Preprocess and add frame to queue
#     processed_frame = preprocess_frame(frame)
#     frame_queue.append(processed_frame)
#
#     # Predict only if we have 20 frames
#     if len(frame_queue) == SEQUENCE_LENGTH:
#         input_sequence = np.expand_dims(np.array(frame_queue), axis=0)  # Shape: (1, 20, 64, 64, 3)
#         predicted_probs = model.predict(input_sequence)[0]  # Get probabilities for both classes
#
#         # Identify class based on threshold
#         predicted_class_name = "crime" if predicted_probs[0] > 0.5 else "non-crime"
#
#         print(f"Predicted Probabilities: {predicted_probs}")
#         print(f"Predicted Class: {predicted_class_name}")
#
#         # Display results on video feed
#         label_text = f"{predicted_class_name}"
#         color = (0, 0, 255) if predicted_class_name == "crime" else (0, 255, 0)
#         cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#
#         # Alert if violence is detected
#         if predicted_class_name == "crime":
#             print("⚠️ ALERT: Violence detected!")
#
#     # Show video feed
#     cv2.imshow("Real-Time Violence Detection (Camo Studio)", frame)
#
#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
