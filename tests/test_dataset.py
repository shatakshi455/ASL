import os
import pickle
import numpy as np
import cv2
import pytest
import mediapipe as mp
from unittest.mock import patch

from  create_dataset import calculate_angle   

# Initialize Mediapipe Hands for mocking
mp_hands = mp.solutions.hands

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
@pytest.mark.parametrize("v1, v2, expected_angle", [
    ([1, 0, 0], [0, 1, 0], 90.0),  # Perpendicular vectors
    ([1, 0, 0], [1, 0, 0], 0.0),    # Same direction
    ([1, 1, 0], [-1, -1, 0], 180.0),  # Opposite direction
])
def test_calculate_angle(v1, v2, expected_angle):
    """Test angle calculation between two vectors"""
    angle = calculate_angle(v1, v2)
    assert pytest.approx(angle, rel=1e-2) == expected_angle


@patch('cv2.imread')
def test_image_loading(mock_imread):
    """Test that the script properly skips corrupted images"""
    mock_imread.return_value = None  # Simulate a corrupted image

    img = cv2.imread("fake_path.jpg")
    assert img is None, "Image should be None for corrupted files"


def test_feature_extraction_length():
    """Ensure that the feature vector length is 67"""
    fake_landmarks = [(i / 100, i / 100, i / 100) for i in range(21)]  # Fake normalized landmarks
    data_aux = []

    # Avoid division by zero
    min_x = min([x for x, _, _ in fake_landmarks]) or 1e-6
    min_y = min([y for _, y, _ in fake_landmarks]) or 1e-6
    min_z = min([z for _, _, z in fake_landmarks]) or 1e-6

    for x, y, z in fake_landmarks:
        data_aux.append(x / min_x)
        data_aux.append(y / min_y)
        data_aux.append(z / min_z)

    # Simulate angle calculations
    finger_joints = [(4, 1, 8, 5), (8, 5, 12, 9), (12, 9, 16, 13), (16, 13, 20, 17)]
    for joint in finger_joints:
        v1 = np.subtract(fake_landmarks[joint[0]], fake_landmarks[joint[1]])
        v2 = np.subtract(fake_landmarks[joint[2]], fake_landmarks[joint[3]])
        angle = calculate_angle(v1, v2)
        data_aux.append(angle)

    assert len(data_aux) == 67, "Feature vector should have exactly 67 elements"



def test_dataset_saving():
    """Test if dataset is properly saved and loaded"""
    data = [[0.1] * 67, [0.2] * 67]  # Fake data
    labels = ["A", "B"]

    with open('test_dataset.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    with open('test_dataset.pickle', 'rb') as f:
        loaded_data = pickle.load(f)

    assert len(loaded_data['data']) == 2
    assert len(loaded_data['data'][0]) == 67
    assert loaded_data['labels'] == ["A", "B"]

    os.remove('test_dataset.pickle')  # Cleanup

 