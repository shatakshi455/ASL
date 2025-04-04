import streamlit as st
st.set_page_config(page_title="ASL Recognition", page_icon="🤟", layout="centered")
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit_extras.switch_page_button as switch

st.sidebar.title("📌 Navigation Menu")
def calculate_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_point(landmark):
    return (landmark.x, landmark.y)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def lines_intersect(p1, p2, q1, q2):
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def check_intersections(landmarks):
    p1, p2 = get_point(landmarks[6]), get_point(landmarks[7])
    q1, q2 = get_point(landmarks[10]), get_point(landmarks[11])

    r1, r2 = get_point(landmarks[7]), get_point(landmarks[8])
    s1, s2 = get_point(landmarks[11]), get_point(landmarks[12])

    t1, t2 = get_point(landmarks[7]), get_point(landmarks[8])
    u1, u2 = get_point(landmarks[10]), get_point(landmarks[11])

    if lines_intersect(p1, p2, q1, q2):
        return 1
    if lines_intersect(r1, r2, s1, s2):
        return 1
    if lines_intersect(t1, t2, u1, u2):
        return 1
    return 0
 
with open('models/model_main.p', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']

# Constants
offset = 20

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

# Labels
labels_dict = {
    0:'space', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
    11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S',
    20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: 'delete'
}


# Streamlit UI
st.title("GestureSpeak: ASL Hand Gesture Recognition")
st.write("Show an ASL hand sign to recognize it. Press **Stop Webcam** to quit.")

# Webcam state
if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False

# Button Controls
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Webcam"):
        st.session_state.run_webcam = True
with col2:
    if st.button("Stop Webcam"):
        st.session_state.run_webcam = False

# Video and output display
stframe = st.empty()
prediction_text = st.empty()
result_text = st.empty()

# Track previous prediction and time
previous_prediction = None
start_time = None

if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = ""   


if st.session_state.run_webcam:
    cap = cv2.VideoCapture(0)

    while st.session_state.run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        predicted_character = "None"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            x_min, y_min = max(0, x_min - offset), max(0, y_min - offset)
            x_max, y_max = min(w, x_max + offset), min(h, y_max + offset)

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
                data_aux.append(y_[i]/ min_y)
                data_aux.append(z_[i]/ min_z)

            finger_joints = [(4, 1, 8, 5), (8, 5, 12, 9), (12, 9, 16, 13), (16, 13, 20, 17)]
            for joint in finger_joints:
                v1 = np.subtract(landmarks[joint[0]], landmarks[joint[1]])
                v2 = np.subtract(landmarks[joint[2]], landmarks[joint[3]])
                data_aux.append(calculate_angle(v1, v2))

            data_aux = np.array(data_aux).reshape(1, -1)

            if data_aux.shape[1] == 67:
                prediction = model.predict(data_aux)
                predicted_index = int(prediction[0])
                predicted_character = labels_dict[predicted_index]

                if(predicted_character == 'M' or predicted_character == 'N'):
                     x = (hand_landmarks.landmark[16].x - hand_landmarks.landmark[12].x)
                     y = (hand_landmarks.landmark[16].y - hand_landmarks.landmark[12].y)

                     with open('models/model_scalerMN.p', 'rb') as f:
                           model_MN = pickle.load(f)['model']
                     
                     
                     X = np.array([[x,y]])
                     pred = model_MN.predict(X)[0]
                     predicted_character = 'M' if pred == '13' else 'N'
                    
                elif(predicted_character == 'K' or predicted_character == 'V' or predicted_character == 'R'):
                                     
                    if(check_intersections(hand_landmarks.landmark) == 1):
                        predicted_character = 'R'
                    else:
                        with open('models/model_scalerKV.p', 'rb') as f:
                           model_KV = pickle.load(f)['model']

                        landmarks = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
                        
                        thumb_tip = landmarks[4]
                        thumb_base = landmarks[2]
                        index_tip = landmarks[8]
                        index_base = landmarks[5]
                        palm_center = landmarks[0]

                        def angle_between_lines(p1, p2, p3, p4):
                            v1 = np.array(p1) - np.array(p2)
                            v2 = np.array(p3) - np.array(p4)
                            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
                            return np.degrees(angle)

                        def same_side(a,  p1, p2):
                            p1x, p1y = p1
                            p2x, p2y = p2
                            ax, ay = a
                            
                            # Compute cross products
                            cross1 = (ay - p1y)*(p2x - p1x) - (p2y - p1y)*(ax - p1x)
                            return cross1 >= 0
                        
                        p1, p2 = landmarks[9], landmarks[10]
                        p3, p4 = landmarks[10], landmarks[11]
                        p5, p6 = landmarks[11], landmarks[12]

                        feature1 = 1 if same_side(thumb_tip, p1, p2) else 0
                        feature2 = 1 if same_side(thumb_tip, p3, p4) else 0
                        feature3 = 1 if same_side(thumb_tip, p5, p6) else 0
                    
                        angle = angle_between_lines(thumb_tip, thumb_base, index_tip, index_base)
                         
                        X = np.array([[angle, feature1, feature2, feature3]])
                        pred = model_KV.predict(X)[0]
                        predicted_character = 'K' if feature1 == 1 else 'V'

                # Track and print if character is stable for 2 seconds
                if predicted_character == previous_prediction:
                    if time.time() - start_time >= 2:
                        if(predicted_character == 'space'):
                            st.session_state.recognized_text+=' '
                        else:
                            if(predicted_character == 'delete'):
                                st.session_state.recognized_text = st.session_state.recognized_text[:-1]
                            else:
                                st.session_state.recognized_text += predicted_character
                        cv2.putText(frame, predicted_character + ' is detected', (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                        previous_prediction = None
                else:
                    previous_prediction = predicted_character
                    start_time = time.time()
            else:
                print("Incorrect number of features")

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        stframe.image(frame, channels="BGR")
        prediction_text.subheader(f"Predicted Character: **{predicted_character}**")
        result_text.subheader(f"Recognized Text: **{st.session_state.recognized_text}**")

    cap.release()
    cv2.destroyAllWindows()
# Display final recognized text after webcam stops
if not st.session_state.run_webcam:
    st.subheader(f"Final Recognized Text: **{st.session_state.recognized_text}**")
