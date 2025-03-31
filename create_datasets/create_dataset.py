import os
import pickle
import cv2
import mediapipe as mp
import numpy as np


# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Data directory
DATA_DIR = 'data'

def calculate_angle(v1, v2):
    """Calculate the angle (in degrees) between two vectors using the dot product formula."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to avoid numerical issues
    return np.degrees(angle)

def create_dataset():
    """Processes images and creates a dataset for sign language detection."""
    data = []
    labels = []

    # Process each class folder
    for dir_ in os.listdir(DATA_DIR):
        counter = 0
        print(f"Processing class: {dir_}")

        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

            data_aux = []  # Reset for each image ✅
            x_, y_, z_ = [], [], []
            landmarks = []

            # Load image
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            if img is None:
                print(f"Skipping corrupted image: {img_path}")
                continue  # Skip corrupted images

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # Process only the first detected hand
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # Only take one hand ✅

                # Extract 3D landmarks
                for i in range(21):  # There are always 21 hand landmarks
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z

                    x_.append(x)
                    y_.append(y)
                    z_.append(z)
                    landmarks.append((x, y, z))

                # Normalize features
                min_x, min_y, min_z = min(x_), min(y_), min(z_)
                for i in range(21):
                    data_aux.append(x_[i] / min_x)
                    data_aux.append(y_[i] / min_y)
                    data_aux.append(z_[i] / min_z)

                # Compute angles between fingers
                finger_joints = [
                    (4, 1, 8, 5),   # Thumb to index
                    (8, 5, 12, 9),  # Index to middle
                    (12, 9, 16, 13), # Middle to ring
                    (16, 13, 20, 17) # Ring to pinky
                ]

                for joint in finger_joints:
                    v1 = np.subtract(landmarks[joint[0]], landmarks[joint[1]])
                    v2 = np.subtract(landmarks[joint[2]], landmarks[joint[3]])
                    angle = calculate_angle(v1, v2)
                    data_aux.append(angle)

                # Ensure exactly 67 features before appending
                if len(data_aux) == 67:
                    data.append(data_aux)
                    labels.append(dir_)
                else:
                    print(f"Skipping {img_path} due to incorrect feature length: {len(data_aux)}")

    # Save dataset
    with open('datasets/dataset_main.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print(f"Dataset saved with {len(data)} samples.")
    return data, labels  # Returning dataset for testing

# Run only if the script is executed directly
if __name__ == "__main__":
    create_dataset()
