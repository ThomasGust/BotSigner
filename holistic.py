import cv2
import mediapipe as mp
import numpy as np

# Initialize the MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Initialize the MediaPipe face detection module
mp_face_detection = mp.solutions.face_detection

# Initialize the MediaPipe holistic module
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
cap = cv2.VideoCapture(0) # replace 0 with the index of your camera if using an external camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(frame_rgb)

    white_frame = np.ones((frame_rgb.shape[0], frame_rgb.shape[1], frame_rgb.shape[2]))*255

    annotated_image = frame.copy()
    mp_drawing.draw_landmarks(annotated_image, results.face_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255)))
    mp_drawing.draw_landmarks(
        annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0)))
    mp_drawing.draw_landmarks(
        annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0)))
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255,0)))
    
    mp_drawing.draw_landmarks(white_frame, results.face_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255)))
    mp_drawing.draw_landmarks(
        white_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0)))
    mp_drawing.draw_landmarks(
        white_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0)))
    mp_drawing.draw_landmarks(
        white_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255,0)))

    # Show the image
    cv2.imshow('MediaPipe Holistic', annotated_image)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite("original_frame.png", frame)
        cv2.imwrite("anottated_image.png", annotated_image)
        cv2.imwrite("white_frame.png", white_frame)
        break

cap.release()
cv2.destroyAllWindows()
