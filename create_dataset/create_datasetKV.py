import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

# --- Mediapipe ---
mp_hands = mp.solutions.hands

# --- Feature Functions ---
def angle_between_lines(p1, p2, p3, p4):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p4)
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def standardised_distance(p1, p2, ref1, ref2):
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    scale = np.linalg.norm(np.array(ref1) - np.array(ref2)) + 1e-6
    return dist / scale

# --- Dataset Directory ---
data_dir = 'data/'

# --- Only use 10 and 21 folders ---
labels = [folder for folder in os.listdir(data_dir) if folder in ['11', '22']]

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
                landmarks = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
                
                # --- Landmarks ---
                thumb_tip = landmarks[4]
                thumb_base = landmarks[2]
                index_tip = landmarks[8]
                index_base = landmarks[5]
                palm_center = landmarks[0]

                # --- Features ---
                angle = angle_between_lines(thumb_tip, thumb_base, index_tip, index_base)
                #thumb_palm_dist = standardised_distance(thumb_tip, palm_center, palm_center, index_base)

                features.append([angle])
                targets.append(label)

print(f"Total Samples Collected: {len(features)}")

# --- Save dataset ---
with open('datasetKV.pickle', 'wb') as f:
    pickle.dump({'data': features, 'labels': targets}, f)

print("Saved features to datasetKV.pickle ✅")
