import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

# --- Mediapipe ---
mp_hands = mp.solutions.hands

# --- Dataset Directory ---
data_dir = 'data/'

# --- Only use 10 and 21 folders ---
labels = [folder for folder in os.listdir(data_dir) if folder in ['13', '14']]

features = []
targets = []

with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    for label in labels:
        folder_path = os.path.join(data_dir, label)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                x = (hand_landmarks.landmark[16].x - hand_landmarks.landmark[12].x)
                y = (hand_landmarks.landmark[16].y - hand_landmarks.landmark[12].y)
                features.append([x, y])
                targets.append(label)

print(f"Total Samples Collected: {len(features)}")

# --- Save dataset ---
with open('datasets/datasetMN.pickle', 'wb') as f:
    pickle.dump({'data': features, 'labels': targets}, f)

print("Saved features to datasetKV.pickle ✅")
