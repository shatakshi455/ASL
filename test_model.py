import pytest
import pickle
import cv2
import mediapipe as mp
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info/warning logs
# Load trained model
with open('./model_scaler.p', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6)

# Labels dictionary
labels_dict = {
    0:'space', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
    11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S',
    20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: 'delete'
}

# Constants
offset = 20

def calculate_angle(v1, v2):
    """ Calculate angle between two vectors """
    v1, v2 = np.array(v1), np.array(v2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

@pytest.mark.parametrize("image_file", os.listdir("asl_alphabet_test/"))
def test_model_prediction(image_file):
    """ Test model predictions on ASL images """

    # Extract ground truth label from filename (before '_')
    expected_label = image_file.split('_')[0]

    # Read image
    img_path = os.path.join("asl_alphabet_test/", image_file)
    img = cv2.imread(img_path)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process image with Mediapipe
    results = hands.process(imgRGB)
    predicted_character = "None"

    if(results.multi_hand_landmarks == None):
        predicted_character = "nothing"
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = img.shape
        x_min, y_min, x_max, y_max = w, h, 0, 0

        # Get bounding box
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)

        x_min, y_min = max(0, x_min - offset), max(0, y_min - offset)
        x_max, y_max = min(w, x_max + offset), min(h, y_max + offset)

        # Extract feature data
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

        # Normalize features
        min_x, min_y, min_z = min(x_), min(y_), min(z_)
        for i in range(21):
            data_aux.append(x_[i]/min_x)
            data_aux.append(y_[i]/min_y)
            data_aux.append(z_[i]/min_z)

        # Compute angles for finger joints
        finger_joints = [(4, 1, 8, 5), (8, 5, 12, 9), (12, 9, 16, 13), (16, 13, 20, 17)]
        for joint in finger_joints:
            v1 = np.subtract(landmarks[joint[0]], landmarks[joint[1]])
            v2 = np.subtract(landmarks[joint[2]], landmarks[joint[3]])
            data_aux.append(calculate_angle(v1, v2))

        # Reshape input for model
        data_aux = np.array(data_aux).reshape(1, -1)

        if data_aux.shape[1] == 67:
            prediction = model.predict(data_aux)
            predicted_index = int(prediction[0])
            predicted_character = labels_dict[predicted_index]

    # Print result (for debugging)
    print(f"File: {image_file}, Expected: {expected_label}, Predicted: {predicted_character}")

    # Assert prediction matches expected label
    assert predicted_character == expected_label, f"Mismatch! Expected {expected_label}, but got {predicted_character}"

if __name__ == "__main__":
    import pytest
    pytest.main(["-v"])
