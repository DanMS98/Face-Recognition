import cv2
import mediapipe as mp
import os

USE_CASCADE = False
USE_LANDMARK = True  #Uses mediapipe mobile net

mp_drawing = mp.solutions.drawing_utils

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
print(cv2_base_dir)
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

faceCascade = cv2.CascadeClassifier(haar_model)

video_capture = cv2.VideoCapture(0)
ret = True
while ret:
    ret, frame = video_capture.read()

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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    elif USE_LANDMARK:
        image_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(image_input)
        if not results.detections:
            print("no face was found")
        else:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()