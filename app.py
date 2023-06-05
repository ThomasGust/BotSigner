import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import mediapipe as mp
import numpy as np
import pandas as pd
from engine.model import load_relevant_data_subset, make_pred, best_n, map_bn

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

vcap = cv2.VideoCapture(0)

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('assets\\asl-icon.png'))
        self.video_df = []
        self.setWindowTitle('BotSign')

        self.video_capture = cv2.VideoCapture(0)  # Open the default camera
        if not self.video_capture.isOpened():
            raise Exception("Could not open video device")

        self.image_label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        self.start_button = QPushButton('Start Video')
        self.start_button.clicked.connect(self.start_video)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop Video')
        self.stop_button.clicked.connect(self.stop_video)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.start_video()

        self.text_box = QTextEdit()
        layout.addWidget(self.text_box)

        self.setLayout(layout)
        self.text_box.setText("")

        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.1)

        self.frame_cntr =0
    def start_video(self):
        self.timer.start(20)  # Update every 20 ms

    def predict_video(self):
        test_df = pd.concat(self.video_df)
        print(test_df.shape)
        _ = test_df.to_parquet("data\\tmp\\CURRENTDATA.parquet")
        data = load_relevant_data_subset("data\\tmp\\CURRENTDATA.parquet")
        preds = make_pred(data)
        print(preds)
        res = best_n(preds, 10)
        print(res)
        out = map_bn(res)
        print(out)
        self.text_box.setText(str(out[::-1]))
    def stop_video(self):
        preds = self.predict_video()
        self.video_df = []
        self.timer.stop()
        self.frame_cntr = 0
    
    def stop(self):
        self.timer.stop()
        self.frame_cntr = 0

    def update_frame(self):
        ret, frame = vcap.read()
        if ret:
            image_orig = frame
            #image = cv2.resize(image_orig, dsize=None, fx=2, fy=2)
            height,width,_ = image_orig.shape
            image_orig.flags.writeable = False

            image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
            result = self.holistic.process(image)
            data = [] 
            fy = height/width
            if result.face_landmarks is None:
                for i in range(468): #
                    data.append({
                        'type' : 'face',
                        'landmark_index' : i,
                        'x' : np.nan,
                        'y' : np.nan,
                        'z' : np.nan,
                    })
            else:
                assert(len(result.face_landmarks.landmark)==468)
                for i in range(468): #
                    xyz = result.face_landmarks.landmark[i]
                    data.append({
                        'type' : 'face',
                        'landmark_index' : i,
                        'x' : xyz.x,
                        'y' : xyz.y *fy,
                        'z' : xyz.z,
                    })
            if result.left_hand_landmarks is None:
                for i in range(21):  #
                    data.append({
                        'type': 'left_hand',
                        'landmark_index': i,
                        'x': np.nan,
                        'y': np.nan,
                        'z': np.nan,
                    })
            else:
                assert (len(result.left_hand_landmarks.landmark) == 21)
                for i in range(21):  #
                    xyz = result.left_hand_landmarks.landmark[i]
                    data.append({
                        'type': 'left_hand',
                        'landmark_index': i,
                        'x': xyz.x,
                        'y': xyz.y *fy,
                        'z': xyz.z,
                    })
            if result.pose_landmarks is None:
                for i in range(33):  #
                    data.append({
                        'type': 'pose',
                        'landmark_index': i,
                        'x': np.nan,
                        'y': np.nan,
                        'z': np.nan,
                    })
            else:
                assert (len(result.pose_landmarks.landmark) == 33)
                for i in range(33):  #
                    xyz = result.pose_landmarks.landmark[i]
                    data.append({
                        'type': 'pose',
                        'landmark_index': i,
                        'x': xyz.x,
                        'y': xyz.y *fy,
                        'z': xyz.z,
                    })
            if result.right_hand_landmarks is None:
                for i in range(21):
                    data.append({
                    'type': 'right_hand',
                    'landmark_index': i,
                    'x': np.nan,
                    'y': np.nan,
                    'z': np.nan,
                    })
            else:
                assert (len(result.right_hand_landmarks.landmark) == 21)
                for i in range(21):
                    xyz = result.right_hand_landmarks.landmark[i]
                    data.append({
                        'type': 'right_hand',
                        'landmark_index': i,
                        'x': xyz.x,
                        'y': xyz.y *fy,
                        'z': xyz.z,
                    })
            annotated_image = image_orig.copy()
            mp_drawing.draw_landmarks(annotated_image, result.face_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255)))
            mp_drawing.draw_landmarks(annotated_image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0)))
            mp_drawing.draw_landmarks(annotated_image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0)))
            mp_drawing.draw_landmarks(annotated_image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255,0)))
            image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            new_width = int(image.shape[1]*3)
            new_height = int(image.shape[0]*3)

            image = cv2.resize(image, (new_width, new_height))
            height, width, _ = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_image))

            frame_df = pd.DataFrame(data)
            frame_df.loc[:,'frame'] =  self.frame_cntr
            frame_df.loc[:, 'height'] = height/width
            frame_df.loc[:, 'width'] = width/width
            self.video_df.append(frame_df)

            self.frame_cntr += 1

    def closeEvent(self, event):
        self.timer.stop()
        self.video_capture.release()
        event.accept()

class HolisticWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('assets\\asl-icon.png'))
        self.video_df = []
        self.setWindowTitle('BotSign')

        self.video_capture = cv2.VideoCapture(0)  # Open the default camera
        if not self.video_capture.isOpened():
            raise Exception("Could not open video device")

        self.image_label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        self.start_button = QPushButton('Start Video')
        self.start_button.clicked.connect(self.start_video)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop Video')
        self.stop_button.clicked.connect(self.stop_video)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.start_video()

        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.1)

        self.frame_cntr =0
    def start_video(self):
        self.timer.start(20)  # Update every 20 ms

    def stop_video(self):
        self.timer.stop()
        self.frame_cntr = 0

    def update_frame(self):
        ret, frame = vcap.read()
        if ret:
            image_orig = frame
            #image = cv2.resize(image_orig, dsize=None, fx=2, fy=2)
            height,width,_ = image_orig.shape
            image_orig.flags.writeable = False
            result = self.holistic.process(image_orig)
            data = [] 
            fy = height/width
            if result.face_landmarks is None:
                for i in range(468): #
                    data.append({
                        'type' : 'face',
                        'landmark_index' : i,
                        'x' : np.nan,
                        'y' : np.nan,
                        'z' : np.nan,
                    })
            else:
                assert(len(result.face_landmarks.landmark)==468)
                for i in range(468): #
                    xyz = result.face_landmarks.landmark[i]
                    data.append({
                        'type' : 'face',
                        'landmark_index' : i,
                        'x' : xyz.x,
                        'y' : xyz.y *fy,
                        'z' : xyz.z,
                    })
            if result.left_hand_landmarks is None:
                for i in range(21):  #
                    data.append({
                        'type': 'left_hand',
                        'landmark_index': i,
                        'x': np.nan,
                        'y': np.nan,
                        'z': np.nan,
                    })
            else:
                assert (len(result.left_hand_landmarks.landmark) == 21)
                for i in range(21):  #
                    xyz = result.left_hand_landmarks.landmark[i]
                    data.append({
                        'type': 'left_hand',
                        'landmark_index': i,
                        'x': xyz.x,
                        'y': xyz.y *fy,
                        'z': xyz.z,
                    })
            if result.pose_landmarks is None:
                for i in range(33):  #
                    data.append({
                        'type': 'pose',
                        'landmark_index': i,
                        'x': np.nan,
                        'y': np.nan,
                        'z': np.nan,
                    })
            else:
                assert (len(result.pose_landmarks.landmark) == 33)
                for i in range(33):  #
                    xyz = result.pose_landmarks.landmark[i]
                    data.append({
                        'type': 'pose',
                        'landmark_index': i,
                        'x': xyz.x,
                        'y': xyz.y *fy,
                        'z': xyz.z,
                    })
            if result.right_hand_landmarks is None:
                for i in range(21):
                    data.append({
                    'type': 'right_hand',
                    'landmark_index': i,
                    'x': np.nan,
                    'y': np.nan,
                    'z': np.nan,
                    })
            else:
                assert (len(result.right_hand_landmarks.landmark) == 21)
                for i in range(21):
                    xyz = result.right_hand_landmarks.landmark[i]
                    data.append({
                        'type': 'right_hand',
                        'landmark_index': i,
                        'x': xyz.x,
                        'y': xyz.y *fy,
                        'z': xyz.z,
                    })
            annotated_image = image_orig.copy()
            mp_drawing.draw_landmarks(annotated_image, result.face_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255)))
            mp_drawing.draw_landmarks(annotated_image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0)))
            mp_drawing.draw_landmarks(annotated_image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0)))
            mp_drawing.draw_landmarks(annotated_image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255,0)))
            image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            new_width = int(image.shape[1]*3)
            new_height = int(image.shape[0]*3)

            image = cv2.resize(image, (new_width, new_height))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_image))

            frame_df = pd.DataFrame(data)
            frame_df.loc[:,'frame'] =  self.frame_cntr
            frame_df.loc[:, 'height'] = height/width
            frame_df.loc[:, 'width'] = width/width
            self.video_df.append(frame_df)

            self.frame_cntr += 1

    def closeEvent(self, event):
        self.timer.stop()
        self.video_capture.release()
        event.accept()

def main():
    app = QApplication([])
    video_window = VideoWindow()
    video_window.show()
    app.exec_()

if __name__ == '__main__':
    main()
