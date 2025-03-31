import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Helper functions remain the same
def dist(l1, l2):
    return ((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)**0.5

def calculate_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Load model
with open('models/model_main.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6)
offset = 20

# Labels dictionary
labels_dict = {
    0:'space', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
    11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S',
    20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: 'delete'
}

def process_image(uploaded_file):
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process image
    results = hands.process(img_rgb)
    predicted_char = "No hand detected"
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = img.shape
        
        # Landmark processing (same as original)
        x_min, y_min, x_max, y_max = w, h, 0, 0
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)
        
        # Feature extraction (same as original)
        data_aux = []
        x_, y_, z_ = [], [], []
        landmarks = []
        
        for i in range(21):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            z = hand_landmarks.landmark[i].z
            x_.append(x)
            y_.append(y)
            z_.append(z)
            landmarks.append((x, y, z))
        
        min_x, min_y, min_z = min(x_), min(y_), min(z_)
        for i in range(21):
            data_aux.append(x_[i]/min_x)
            data_aux.append(y_[i]/min_y)
            data_aux.append(z_[i]/min_z)
        
        # Angle calculation
        finger_joints = [(4, 1, 8, 5), (8, 5, 12, 9), (12, 9, 16, 13), (16, 13, 20, 17)]
        for joint in finger_joints:
            v1 = np.subtract(landmarks[joint[0]], landmarks[joint[1]])
            v2 = np.subtract(landmarks[joint[2]], landmarks[joint[3]])
            data_aux.append(calculate_angle(v1, v2))
        
        data_aux = np.array(data_aux).reshape(1, -1)
        
        # Prediction
        if data_aux.shape[1] == 67:
            prediction = model.predict(data_aux)
            predicted_index = int(prediction[0])
            predicted_char = labels_dict[predicted_index]

            # Special handling for K/V
            if predicted_char in ['K', 'V']:
                with open('models/model_scalerKV.p', 'rb') as f:
                    model_KV = pickle.load(f)['model']
                
                data_auxkv = [dist(landmarks[i], landmarks[4])/min_z 
                            for i in range(21) if i != 4]
                data_auxkv = np.array(data_auxkv).reshape(1, -1)
                
                if data_auxkv.shape[1] == 20:
                    prediction_kv = model_KV.predict(data_auxkv)
                    predicted_char = labels_dict[int(prediction_kv[0])]

        # Draw annotations
        x_min, y_min = max(0, x_min - offset), max(0, y_min - offset)
        x_max, y_max = min(w, x_max + offset), min(h, y_max + offset)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, predicted_char, (x_min, y_min - 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    
    return img, predicted_char

# Streamlit UI
st.title("GestureSpeak: ASL Image Recognition")
uploaded_file = st.file_uploader("Upload ASL hand image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    processed_img, prediction = process_image(uploaded_file)
    st.image(processed_img, channels="BGR", caption="Processed Image")
    st.subheader(f"Predicted Character: {prediction}")