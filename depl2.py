import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class ASLTransformer(VideoTransformerBase):
    def __init__(self, model, labels_dict):
        self.model = model
        self.labels_dict = labels_dict
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
        self.offset = 20
        self.previous_prediction = None
        self.start_time = None
        self.recognized_text = ""

    def calculate_angle(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        predicted_character = "None"
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = img.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            x_min, y_min = max(0, x_min - self.offset), max(0, y_min - self.offset)
            x_max, y_max = min(w, x_max + self.offset), min(h, y_max + self.offset)

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

            finger_joints = [(4, 1, 8, 5), (8, 5, 12, 9), (12, 9, 16, 13), (16, 13, 20, 17)]
            for joint in finger_joints:
                v1 = np.subtract(landmarks[joint[0]], landmarks[joint[1]])
                v2 = np.subtract(landmarks[joint[2]], landmarks[joint[3]])
                data_aux.append(self.calculate_angle(v1, v2))

            data_aux = np.array(data_aux).reshape(1, -1)

            if data_aux.shape[1] == 67:
                prediction = self.model.predict(data_aux)
                predicted_index = int(prediction[0])
                predicted_character = self.labels_dict[predicted_index]

                # Track and print if character is stable for 2 seconds
                if predicted_character == self.previous_prediction:
                    if time.time() - self.start_time >= 2:
                        if predicted_character == 'space':
                            self.recognized_text += ' '
                        elif predicted_character == 'delete':
                            self.recognized_text = self.recognized_text[:-1]
                        else:
                            self.recognized_text += predicted_character
                        self.previous_prediction = None
                else:
                    self.previous_prediction = predicted_character
                    self.start_time = time.time()

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img, predicted_character, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        st.subheader(f"Predicted Character: **{predicted_character}**")
        st.subheader(f"Recognized Text: **{self.recognized_text}**")
        return img

# Load trained model
with open('./model_scaler.p', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']

labels_dict = {
    0:'space', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 
    10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S',
    20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: 'delete'
}

st.title("GestureSpeak: ASL Hand Gesture Recognition")
webrtc_streamer(key="asl", video_transformer_factory=lambda: ASLTransformer(model, labels_dict))
