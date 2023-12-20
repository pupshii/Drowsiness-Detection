#test.py   CNN for video for driver status detection
import cv2
import dlib
import numpy as np
import scipy
import keras
import os
import h5py
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator


data = []
labels = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load closed eye images
closed_path = 'Closed'
for file_name in os.listdir(closed_path):
    file_path = os.path.join(closed_path, file_name)
    image = cv2.imread(file_path)
    image = cv2.resize(image, (320, 240))  # Resize image to a desired size
    data.append(image)
    labels.append(0)  # Assign a label of 0 for closed eyes

# Load open eye images
open_path = 'Open'
for file_name in os.listdir(open_path):
    file_path = os.path.join(open_path, file_name)
    image = cv2.imread(file_path)
    image = cv2.resize(image, (320, 240))  # Resize image to a desired size
    data.append(image)
    labels.append(1)  # Assign a label of 1 for open eyes

# Convert the data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode the labels into numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Normalize the pixel values of the images
train_data = train_data / 255.0
test_data = test_data / 255.0

# Reshape the training data
train_data = np.transpose(train_data, (0, 2, 1, 3))
test_data = np.transpose(test_data, (0, 2, 1, 3))


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(320, 240, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=5, batch_size=32)

test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

model.summary() 

video_path = '/Users/pupshi/Downloads/KU/Video/Video.mp4'  # Replace 'your_video_path' with the path to your input video file
output_path = '/Users/pupshi/Downloads/KU/Video/output.mp4'  # Replace 'your_output_path' with the desired path for the output video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize the output video writer (optional)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for the output video
output_video = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))

# VV Video Prediction VV

# Preprocess the frame (e.g., resize, normalize, etc.) to match the input shape of the CNN model
def preprocess_frame(frame):
    #resized_frame = cv2.resize(frame, (640, 480))
    normalized_frame = frame / 255.0
    resized_frame = cv2.resize(normalized_frame, (320, 240))
    return resized_frame


def predict_eye_status(frame):
    frame = preprocess_frame(frame)
    
    frame = np.expand_dims(frame, axis=0)
    # Convert the processed frame to the appropriate data type
    frame = (frame * 255).astype(np.uint8)
    # Transpose the processed frame to match the expected input shape of the CNN model
    frame = np.transpose(frame, (0, 2, 1, 3))

    prediction = model.predict(frame)
    print("Prediction:", prediction)
    status = "Open" if prediction[0][0] > 0.5 else "Closed"
    return status

def detect_eyes(frame):

    rx1 = 0
    rx2 = 0
    ry1 = 0
    ry2 = 0
    lx1 = 0
    lx2 = 0
    ly1 = 0
    ly2 = 0
    status_right = None
    status_left = None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the face inside of the frame
    faces = detector(gray)

    # Iterate faces.
    for face in faces:

        face_coordinates = predictor(gray, face)

        rx1 = face_coordinates.part(37).x-35
        rx2 = face_coordinates.part(40).x+35
        ry1 = face_coordinates.part(20).y+10
        ry2 = face_coordinates.part(42).y+40

        lx1 = face_coordinates.part(43).x-35
        lx2 = face_coordinates.part(46).x+35
        ly1 = face_coordinates.part(25).y+10
        ly2 = face_coordinates.part(47).y+40
        cv2.rectangle(frame, (rx1, ry1),
                     (rx2, ry2), (0, 255, 0), 2)
        cv2.rectangle(frame, (lx1, ly1),
                     (lx2, ly2), (0, 255, 0), 2)

    status_right = predict_eye_status(frame[ry1:ry2,rx1:rx2])
    status_left = predict_eye_status(frame[ly1:ly2,lx1:lx2]) 
    cv2.imshow('Frame Right', frame[ry1:ry2,rx1:rx2])
    cv2.imshow('Frame Left', frame[ly1:ly2,lx1:lx2])

    print("Right:", status_right)
    print("Left:", status_left)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    detect_eyes(frame)
     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()