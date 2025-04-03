import pytest
import os
import cv2
import numpy as np  
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pages.Check_From_Images import process_image 
TEST_DIR = "asl_alphabet_test"

@pytest.mark.parametrize("image_file", os.listdir(TEST_DIR))
def test_asl_prediction(image_file):
    expected_label = image_file.split('_')[0]  # Extracts expected label from filename
    image_path = os.path.join(TEST_DIR, image_file)
    
    with open(image_path, "rb") as img_file:
        processed_img, predicted_label = process_image(img_file)
    
    assert predicted_label == expected_label, f"Failed for {image_file}: Expected {expected_label}, but got {predicted_label}"
