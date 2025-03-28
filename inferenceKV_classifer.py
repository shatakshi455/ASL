import cv2
import mediapipe as mp
import numpy as np
import pickle

# --- Load Trained Model ---
with open('model_scalerKV.p', 'rb') as f:
    model = pickle.load(f)['model']

# --- Mediapipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Feature Functions ---
def angle_between_lines(p1, p2, p3, p4):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p4)
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def normalized_distance(p1, p2, scale):
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    return dist / (scale + 1e-6)

# --- Inference ---
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(lm.x * w, lm.y * h, lm.z) for lm in hand_landmarks.landmark]
                
                # --- Feature Extraction ---
                thumb_tip = landmarks[4]
                thumb_base = landmarks[2]
                index_tip = landmarks[8]
                index_base = landmarks[5]
                palm_center = landmarks[0]

                angle = angle_between_lines(thumb_tip, thumb_base, index_tip, index_base)
                scale = abs(palm_center[2]) + 1e-6
                thumb_palm_dist = normalized_distance(thumb_tip[:2], palm_center[:2], scale)

                # --- Prediction ---
                X = np.array([[angle]])
                pred = model.predict(X)[0]

                label = 'K' if pred == '10' else 'V'

                # --- Display Values ---
                cv2.putText(frame, f"Angle: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"Thumb-Palm Dist: {thumb_palm_dist:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(frame, f"Prediction: {label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Sign Detection with Features", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
