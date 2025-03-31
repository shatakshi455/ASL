import cv2
import mediapipe as mp
import numpy as np

# Init MediaPipe
mp_hands = mp.solutions.hands

# Capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape

                # Get landmark points
                landmark_points = np.array(
                    [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark])

                # Get rectangular bounding box
                x_min = np.min(landmark_points[:, 0]) - 20
                y_min = np.min(landmark_points[:, 1]) - 20
                x_max = np.max(landmark_points[:, 0]) + 20
                y_max = np.max(landmark_points[:, 1]) + 20

                # Clip bounding box to image size
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                # Crop rectangle region
                hand_crop = frame[y_min:y_max, x_min:x_max].copy()

                # Create mask for cropped region only
                shifted_points = landmark_points - np.array([x_min, y_min])  # shift to crop coordinates
                mask = np.zeros(hand_crop.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [shifted_points], 255)

                # Remove background inside the rectangle
                hand_only = cv2.bitwise_and(hand_crop, hand_crop, mask=mask)

                # Grayscale & Smooth
                gray = cv2.cvtColor(hand_only, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                # Edge Detection
                edges = cv2.Canny(gray, threshold1=50, threshold2=150)
                sketch = cv2.bitwise_not(edges)

                # Show final
                hand_crop = cv2.Canny(hand_crop, threshold1=50, threshold2=150)
                cv2.imshow("Rectangular Cropped Hand Sketch", hand_crop)

        else:
            cv2.imshow("Rectangular Cropped Hand Sketch", np.ones_like(frame) * 255)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
