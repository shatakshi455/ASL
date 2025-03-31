import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Data directory
DATA_DIR = './data'


def calculate_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def dist(l1, l2):
    return ((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)**0.5


def create_dataset():
    data = []
    labels = []

    # Only process '11' and '22' folders
    for dir_ in ['11', '22']:
        print(f"Processing class: {dir_}")

        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []
            x_, y_, z_ = [], [], []
            landmarks = []

            # Load image
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            if img is None:
                print(f"Skipping corrupted image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                for i in range(21):
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
                    if i != 4:
                        data_aux.append(dist(landmarks[i], landmarks[4]) / min_z)

                # Print the data_aux for the current image
                print(f"{dir_}data_aux for {img_path}: {data_aux[0]}")

                if len(data_aux) == 20:
                    data.append(data_aux)
                    labels.append(dir_)
                else:
                    print(f"Skipping {img_path} due to incorrect feature length: {len(data_aux)}")
            else:
                print(f"No hand landmarks found in {img_path}")

    # Save dataset
    with open('datasetKV.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print(f"Dataset saved with {len(data)} samples.")
    return data, labels


if __name__ == "__main__":
    create_dataset()
