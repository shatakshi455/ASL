import cv2
import mediapipe as mp
import numpy as np

# Init MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = np.ones_like(frame) * 255  # white background

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw custom lines (like the sketch)
                landmark_points = []
                h, w, _ = image.shape
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmark_points.append((x, y))

                # Draw connections
                connections = mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx, end_idx = connection
                    x1, y1 = landmark_points[start_idx]
                    x2, y2 = landmark_points[end_idx]
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 3)  # thick black lines

                # Draw keypoints as small circles (optional)
                for point in landmark_points:
                    cv2.circle(image, point, 5, (0, 0, 0), -1)

        cv2.imshow("ASL Style Drawing", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
