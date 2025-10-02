import os
import sys
import cv2
import mediapipe as mp

# Determine the base directory
if getattr(sys, 'frozen', False):
    # If the application is running as a bundled executable
    base_dir = sys._MEIPASS
else:
    # If running as a script
    base_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize the mediapipe hands module.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam.
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    """
    Count the number of fingers that are up, based on hand landmarks.
    """
    fingers = []

    # Thumb (using x coordinates because thumb moves horizontally)
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(1)  # Thumb is open
    else:
        fingers.append(0)  # Thumb is closed

    # Other four fingers (using y coordinates because they move vertically)
    for i in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
              mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
              mp_hands.HandLandmark.RING_FINGER_TIP,
              mp_hands.HandLandmark.PINKY_TIP]:
        if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y:
            fingers.append(1)  # Finger is open
        else:
            fingers.append(0)  # Finger is closed

    return fingers.count(1)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the BGR image to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands.
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count the number of fingers.
            finger_count = count_fingers(hand_landmarks)
            print(f"Hand {idx + 1}: Number of fingers: {finger_count}")

            # Display the finger count on the frame.
            cv2.putText(frame, f'Hand {idx + 1}: {finger_count} fingers', 
                        (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame with the drawn landmarks and finger count.
    cv2.imshow('Hand Tracking', frame)

    # Break the loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources.
cap.release()
cv2.destroyAllWindows()
