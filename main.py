import time

import cv2
import mediapipe as mp
import os

USE_CASCADE = False  # Uses classic haar solution
USE_LANDMARK = True  # Uses mediapipe mobile net
DISPLAY_TIME = 1  # Updates FPS. sec

mp_drawing = mp.solutions.drawing_utils

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

faceCascade = cv2.CascadeClassifier(haar_model)

prev_time = time.time()

video_capture = cv2.VideoCapture(0)
ret = True
frame_counter = 0
disp = "FPS: "

while ret:
    ret, frame = video_capture.read()
    frame_counter += 1
    if USE_CASCADE:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            # flags=
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    elif USE_LANDMARK:
        image_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(image_input)
        if not results.detections:
            print("no face was found")
        else:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

    time_dif = time.time() - prev_time
    if time_dif >= DISPLAY_TIME:
        fps = frame_counter/time_dif
        frame_counter = 0
        disp = "FPS: "+str(fps)[:5]
        prev_time = time.time()

    cv2.putText(frame, disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
