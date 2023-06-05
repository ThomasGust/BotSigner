from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import mediapipe as mp
import time

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
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

            # Convert the image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                # Use MediaPipe Holistic to extract keypoints
                results = holistic.process(frame_rgb)

                # Access the keypoints
                # Example: Get the position of the nose
                nose_x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x
                nose_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y

                # Draw the keypoints on the image
                annotated_image = frame.copy()
                mp_drawing.draw_landmarks(annotated_image, results.face_landmarks)
                mp_drawing.draw_landmarks(
                    annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            except Exception as e:
                annotated_image = frame_rgb
            self.change_pixmap_signal.emit(annotated_image)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt live label demo")
        self.disply_width = 640*5
        self.display_height = 480*5
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Webcam')

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()



    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())