import cv2
import mediapipe as mp
import numpy as np

# --- Mediapipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Feature Functions ---
def angle_between_lines(p1, p2, p3, p4):
    """Angle between line(p1,p2) and line(p3,p4)"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p4)
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def normalized_distance(p1, p2):
    """Normalized Euclidean distance between two 3D points"""
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    # Normalize by the hand scale (distance between wrist (0) and middle finger MCP (9) as reference)
    scale = np.linalg.norm(np.array(p1) - np.array(p2)) if np.linalg.norm(np.array(p1) - np.array(p2)) != 0 else 1
    return dist / scale

# --- Webcam ---
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
                # --- Get full landmark with z ---
                landmarks = [(lm.x * w, lm.y * h, lm.z) for lm in hand_landmarks.landmark]
                
                # --- Points ---
                thumb_tip = landmarks[4]
                thumb_base = landmarks[2]
                index_tip = landmarks[8]
                index_base = landmarks[5]
                palm_center = landmarks[0]
                ref_point = landmarks[9]  # Use middle finger MCP as hand-size reference
                
                # --- Feature 1: Angle ---
                angle = angle_between_lines(thumb_tip, thumb_base, index_tip, index_base)

                # --- Feature 2: Normalized Distance ---
                hand_scale = np.linalg.norm(np.array(palm_center) - np.array(ref_point)) + 1e-6  # avoid zero division
                thumb_palm_dist = np.linalg.norm(np.array(thumb_tip) - np.array(palm_center)) / hand_scale
                
                # --- Display Features ---
                cv2.putText(frame, f"Angle: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                cv2.putText(frame, f"Norm Dist: {thumb_palm_dist:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                # --- Draw landmarks ---
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Features", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
