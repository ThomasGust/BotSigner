'''
pip install mediapipe==0.9.0.1 
'''
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

video_file = '/home/titanx/Downloads/cat.mp4'
cap = cv2.VideoCapture('YES.mp4')

def generate_keypoints(cap, display=True):
	holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.1)

	video_df = []
	frame_no=0
	while cap.isOpened():
		print('\r',frame_no,end='')
		success, image_orig = cap.read()

		if not success: break
		image = cv2.resize(image_orig, dsize=None, fx=4, fy=4)
		height,width,_ = image.shape

		#print(image.shape)
		image.flags.writeable = False
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		result = holistic.process(image)

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
			for i in range(468):
				xyz = result.face_landmarks.landmark[i]
				data.append({
					'type' : 'face',
					'landmark_index' : i,
					'x' : xyz.x,
					'y' : xyz.y *fy,
					'z' : xyz.z,
				})
		if result.left_hand_landmarks is None:
			for i in range(21):
				data.append({
					'type': 'left_hand',
					'landmark_index': i,
					'x': np.nan,
					'y': np.nan,
					'z': np.nan,
				})
		else:
			assert (len(result.left_hand_landmarks.landmark) == 21)
			for i in range(21):
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

		# -----------------------------------------------------
		if result.right_hand_landmarks is None:
			for i in range(21):  #
				data.append({
					'type': 'right_hand',
					'landmark_index': i,
					'x': np.nan,
					'y': np.nan,
					'z': np.nan,
				})
		else:
			assert (len(result.right_hand_landmarks.landmark) == 21)
			for i in range(21):  #
				xyz = result.right_hand_landmarks.landmark[i]
				data.append({
					'type': 'right_hand',
					'landmark_index': i,
					'x': xyz.x,
					'y': xyz.y *fy,
					'z': xyz.z,
				})
			zz=0

		frame_df = pd.DataFrame(data)
		frame_df.loc[:,'frame'] =  frame_no
		frame_df.loc[:, 'height'] = height/width
		frame_df.loc[:, 'width'] = width/width
		video_df.append(frame_df)
		#if frame_no >= 64:
			#break


		#=========================
		frame_no +=1

		if display:
			annotated_image = image_orig.copy()
			mp_drawing.draw_landmarks(annotated_image, result.face_landmarks,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255)))
			mp_drawing.draw_landmarks(
				annotated_image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0)))
			mp_drawing.draw_landmarks(
				annotated_image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0)))
			mp_drawing.draw_landmarks(
				annotated_image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255,0)))
			
			cv2.imshow("Annotated Image", annotated_image)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
		
	cv2.destroyAllWindows()


	video_df = pd.concat(video_df)
	holistic.close()
	data = video_df.to_parquet('video_df.parquet')
	return video_df

def keypoints_from_camera():
	cap = cv2.VideoCapture(0)

	return generate_keypoints(cap)
#print(data)