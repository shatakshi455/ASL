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

def same_side(a,  p1, p2):
    """
    Check if points a and b lie on the same side of the line formed by points p1 and p2 using x and y coordinates.
    """
    p1x, p1y = p1
    p2x, p2y = p2
    ax, ay = a
    
    # Compute cross products
    cross1 = (ay - p1y)*(p2x - p1x) - (p2y - p1y)*(ax - p1x)
    return cross1 >= 0

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

                # --- Line Landmarks ---
                p1, p2 = landmarks[9], landmarks[10]
                p3, p4 = landmarks[10], landmarks[11]
                p5, p6 = landmarks[11], landmarks[12]

                # --- Features ---
                angle = angle_between_lines(thumb_tip, thumb_base, index_tip, index_base)

                # --- Additional Features ---
                feature1 = 1 if same_side(thumb_tip, p1, p2) else 0
                feature2 = 1 if same_side(thumb_tip, p3, p4) else 0
                feature3 = 1 if same_side(thumb_tip, p5, p6) else 0
               
                print(label, angle, feature1, feature2, feature3)
                features.append([angle, feature1, feature2, feature3])
                targets.append(label)

print(f"Total Samples Collected: {len(features)}")

# --- Save dataset ---
with open('datasets/datasetKV.pickle', 'wb') as f:
    pickle.dump({'data': features, 'labels': targets}, f)

print("Saved features to datasetKV.pickle ✅")
