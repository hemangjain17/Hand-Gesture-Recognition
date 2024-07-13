import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Constants
GESTURE_CATEGORIES = {'Backward': 0, 'Down':1, 'Flip':2, 'Forward':3, 'Land':4, 'Left':5, 'RIght':6, 'Up':7}
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720  # Adjust based on your requirements
DATA_DIR = "C:/First/hand gestures"
CSV_FILENAME = "hand_landmarks_datasets.csv"
MODEL_FILENAME = "hand_gesture_model.h5"

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    print(f"Error: Dataset directory '{DATA_DIR}' not found.")
    exit()

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

# Load images, extract hand landmarks, and save data to CSV
data = {"landmarks": [], "label": []}

for label in (GESTURE_CATEGORIES):
    category_dir = os.path.join(DATA_DIR, str(label))

    if not os.path.exists(category_dir):
        print(f"Error: Category directory '{category_dir}' not found.")
        exit()

    for image_name in os.listdir(category_dir):
        image_path = os.path.join(category_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        landmarks = extract_hand_landmarks(image)
        data["landmarks"].append(landmarks)
        data["label"].append(GESTURE_CATEGORIES[label])

# Convert the data dictionary to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.dropna(inplace=True)
df.to_csv(CSV_FILENAME, index=False)
print(f"Hand landmarks data saved to '{CSV_FILENAME}'.")


