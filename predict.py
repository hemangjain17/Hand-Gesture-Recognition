import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from tensorflow import keras
import json
GESTURE_CATEGORIES = {'Backward': 0, 'Cheers':1, 'Flip':2, 'Forward':3, 'Land':4, 'Left':5, 'RIght':6, 'Up':7 , "Invalid Gesture" :8 }
# Constants
IMAGE_WIDTH, IMAGE_HEIGHT = 1280 ,720


# Load the saved Keras model
MODEL_FILENAME = "hand_gesture_model.h5"
loaded_model = keras.models.load_model(MODEL_FILENAME)

# Function to extract hand landmarks using Mediapipe
def extract_hand_landmarks(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe Hands
    results = hands.process(rgb_image)
    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for point in hand_landmarks.landmark:
                landmarks.extend([point.x, point.y, point.z])

    hands.close()
    return landmarks

# Initialize OpenCV VideoCapture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize the frame
    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Extract hand landmarks
    # Extract hand landmarks
    landmarks = extract_hand_landmarks(frame)
    if not landmarks:
        print("No hand landmarks detected.")
        continue
    # Process landmarks for prediction
    landmarks = np.array([landmarks])
    landmarks = landmarks.reshape(landmarks.shape[0], 21, 3)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(landmarks)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(landmarks)

    # Get the predicted class
    predicted_class = np.argmax(predictions)
    for key in GESTURE_CATEGORIES :
        if (GESTURE_CATEGORIES[key]==predicted_class):

    # Display the predicted class on the frame
            cv2.putText(frame, f'Gesture: {key}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
