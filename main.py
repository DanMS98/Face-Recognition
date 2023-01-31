import time

import cv2
import numpy as np
import mediapipe as mp


def mp_to_pix(mp_x, mp_y, mp_width, mp_height, frame_width, frame_heigth):
    x, y = int(mp_x * frame_width), int(mp_y * frame_heigth)
    width, height = int(mp_width * frame_w), int(mp_height * frame_h)
    return x, y, width, height

USE_CASCADE = False  # Uses classic haar solution
USE_LANDMARK = True  # Uses mediapipe mobile net
DISPLAY_TIME = 1  # Updates FPS. (sec)
ROI_SIZE = (255, 255)

if USE_LANDMARK:
    mp_drawing = mp.solutions.drawing_utils
    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

if USE_CASCADE:
    import os
    cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(haar_model)

prev_time = time.time()

video_capture = cv2.VideoCapture(0)
frame_counter = 0
disp = "FPS: "

ret, frame = video_capture.read()
frame_h, frame_w, frame_c = frame.shape

while ret:
    face_roi = np.zeros(ROI_SIZE)
    face_frames = []
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
            disp = "No Face Detected"
        else:
            for detection in results.detections:
                relative_bounding_box = detection.location_data.relative_bounding_box
                x, y, width, height = mp_to_pix(relative_bounding_box.xmin,
                                                relative_bounding_box.ymin,
                                                relative_bounding_box.width,
                                                relative_bounding_box.height,
                                                frame_w, frame_h)

                face_roi = frame[y:y+height, x:x+width]
                face_roi = cv2.resize(face_roi, ROI_SIZE)
                face_frames.append(face_roi)
                mp_drawing.draw_detection(frame, detection)

    time_dif = time.time() - prev_time
    if time_dif >= DISPLAY_TIME:
        fps = frame_counter/time_dif
        frame_counter = 0
        disp = "FPS: "+str(fps)[:5]
        prev_time = time.time()

    cv2.putText(frame, disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imshow('Video', frame)
    if len(face_frames) > 0:
        cv2.imshow('roi', face_roi)
    else:
        if cv2.getWindowProperty('roi', cv2.WND_PROP_VISIBLE) == 1:
            cv2.destroyWindow('roi')

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('p'):
        cv2.waitKey(-1)
    # elif k == ord('s'):
    #     print("Saving...")
    #     cv2.waitKey(-1)

    ret, frame = video_capture.read()

video_capture.release()
cv2.destroyAllWindows()
