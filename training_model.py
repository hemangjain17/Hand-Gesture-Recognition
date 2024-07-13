import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers , callbacks
#['Backward', 'Down', 'Flip', 'Forward', 'Land', 'Left', 'RIght', 'Up']
# Constants
GESTURE_CATEGORIES = 8
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
print(f"Reading ... '{CSV_FILENAME}'.")
df = pd.read_csv(CSV_FILENAME)



# Extract features (landmarks) and labels

X = np.array([np.fromstring(x[1:-1], sep=',', dtype=float).reshape(-1, 3) for x in df['landmarks']])


y = df['label'].values


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)


# print(len(X_train[0]))
# Define the Keras model
model = keras.models.Sequential([
    keras.layers.Input(shape=(21, 3), dtype='float32'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dropout(0.5),
    keras.layers.Dense(8, activation='softmax')
])

X_train = X_train.astype(float)
X_test = X_test.astype(float)
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[early_stopping])
# Train the model
# model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

# Save the trained model
model.save(MODEL_FILENAME)
print(f"Model saved to '{MODEL_FILENAME}'.")

