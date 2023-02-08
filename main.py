import time
import cv2
import numpy as np
import mediapipe as mp
from argparse import ArgumentParser
from Face2Encoding import normalize
from inceptionresnet import InceptionResNetV2
import pickle
from scipy.spatial.distance import cosine


class FaceDetector:
    def __init__(self, path_to_weights, path_to_encodings_database, use_cascade=False):
        self.path_to_model_weights = path_to_weights
        self.path_to_encodings_dictionary = path_to_encodings_database
        self.ROI_SIZE = (255, 255)
        self.MIN_FACE_THRESHOLD = 0.6
        self.DISPLAY_TIME = 1  # Updates FPS. (sec)
        self.using_cascade = False
        self.using_ssd = True
        if use_cascade:
            self.using_cascade = True
            import os
            cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
            haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
            self.faceCascade = cv2.CascadeClassifier(haar_model)
        else:
            self.using_ssd = True
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.9
            )
        self.mp_hand = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self._load_encoder_and_encodings()

    def _load_encoder_and_encodings(self):
        self.model = InceptionResNetV2()
        self.model.load_weights(self.path_to_model_weights)
        with open(self.path_to_encodings_dictionary, 'rb') as f:
            self.encoding_dict = pickle.load(f)

    def _cascade_process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_dict = {}
        faces_from_cascade = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            # flags=
        )

        for (x, y, w, h) in faces_from_cascade:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = frame[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, self.ROI_SIZE)
            faces_dict[(x, y)] = face_roi
        return frame, faces_dict

    def _mp_to_pix(self, mp_x, mp_y, mp_width, mp_height, frame_width, frame_heigth):
        x, y = int(mp_x * frame_width), int(mp_y * frame_heigth)
        width, height = int(mp_width * frame_width), int(mp_height * frame_heigth)
        return x, y, width, height

    def _mp_process(self, frame):
        faces = {}
        frame_h, frame_w, frame_c = frame.shape
        image_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(image_input)
        if results.detections:
            for detection in results.detections:
                relative_bounding_box = detection.location_data.relative_bounding_box
                x, y, width, height = self._mp_to_pix(relative_bounding_box.xmin,
                                                relative_bounding_box.ymin,
                                                relative_bounding_box.width,
                                                relative_bounding_box.height,
                                                frame_w, frame_h)

                face_roi = frame[y:y + height, x:x + width]
                face_roi = cv2.resize(face_roi, self.ROI_SIZE)
                faces[(x, y)] = face_roi
                self.mp_drawing.draw_detection(frame, detection)
        return frame, faces

    def detect_from_video(self):
        prev_time = time.time()
        video_capture = cv2.VideoCapture(0)
        frame_counter = 0
        disp = "FPS: "
        ret, frame = video_capture.read()
        while ret:
            faces = {}
            frame_counter += 1

            # hand_results = self.mp_hand.process(frame)

            if self.using_cascade:
                frame, faces = self._cascade_process(frame)

            elif self.using_ssd:
                frame, faces = self._mp_process(frame)
                if len(faces) == 0:
                    disp = "No Face Detected"

            for loc, face in faces.items():
                face = normalize(face)
                face = cv2.resize(face, (160, 160))
                face_encoding = self.model.predict(np.expand_dims(face, axis=0))[0]
                distance = float("inf")
                name = 'unknown'
                for db_name, db_encode in self.encoding_dict.items():
                    dist = cosine(db_encode, face_encoding)
                    if dist < self.MIN_FACE_THRESHOLD and dist < distance:
                        name = db_name
                        distance = dist
                if name == 'unknown':
                    cv2.putText(frame, name, loc, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    cv2.putText(frame, name, loc, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # if hand_results.multi_hand_landmarks:
            #     for hand_landmarks in hand_results.multi_hand_landmarks:
            #         self.mp_drawing.draw_landmarks(
            #             frame,
            #             hand_landmarks,
            #             # self.mp_hand.HAND_CONNECTIONS,
            #         )

            time_dif = time.time() - prev_time
            if time_dif >= self.DISPLAY_TIME:
                fps = frame_counter / time_dif
                frame_counter = 0
                disp = "FPS: " + str(fps)[:5]
                prev_time = time.time()

            cv2.putText(frame, disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 255, 100), 1)
            cv2.imshow('Video', frame)
            if len(faces) > 0:
                cv2.imshow('roi', list(faces.values())[0])
            else:
                if cv2.getWindowProperty('roi', cv2.WND_PROP_VISIBLE) == 1:
                    cv2.destroyWindow('roi')

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('p'):
                cv2.waitKey(-1)
            # elif k == ord('s'):
            #     cv2.waitKey(-1)
            ret, frame = video_capture.read()
        video_capture.release()
        cv2.destroyAllWindows()

    def detect_from_image(self):
        path = args.pic_path
        # print(path)
        frame = cv2.imread(path)
        if self.using_cascade:
            frame, faces = self.cascade_process(frame)

        elif self.using_ssd:
            frame, faces = self._mp_process(frame)

            cv2.imshow("output", frame)
            cv2.waitKey(0)
            i = 0
            for loc, face in faces.items():
                filename = f'face{i}.jpg'
                i = i+1
                cv2.imshow(filename, face)
                cv2.imwrite(filename, face)
                cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--pic_path', type=str, default="pic.jpg")
    parser.add_argument('--use_vid', type=int, default=1)
    args = parser.parse_args()

    USE_VIDEO = True if args.use_vid == 1 else False
    USE_CASCADE = False  # Uses classic haar solution, otherwise uses mediapipe mobile net

    detector = FaceDetector('facenet_keras_weights.h5', 'encodings/encodings.pkl', USE_CASCADE)
    if USE_VIDEO:
        detector.detect_from_video()
    else:
        detector.detect_from_image()
