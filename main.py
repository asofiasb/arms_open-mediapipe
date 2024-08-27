import cv2
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

def arms_open(landmarks):
    
    elbow_left = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])

    elbow_right = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])

    hip_right = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])

    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])

    distBracoEsquerdo_toQuadril = np.linalg.norm(elbow_left - left_hip)
    distBracoDireito_toQuadril = np.linalg.norm(elbow_right - hip_right)

    print(f"Distância do braço esquerdo ao quadril: {distBracoEsquerdo_toQuadril}")
    print(f"Distância do braço direito ao quadril: {distBracoDireito_toQuadril}")

    return distBracoEsquerdo_toQuadril > 0.3 and distBracoDireito_toQuadril > 0.3

while cap.isOpened():
    success, image = cap.read()

    if not success:
        print("Ignoring empty frame.")
        continue

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        if arms_open(landmarks):
            cv2.putText(image, "Arms Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "Arms Not Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('MediaPipe Pose Recognition', image)

    #matar o terminal com 'q' ou control+C
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
