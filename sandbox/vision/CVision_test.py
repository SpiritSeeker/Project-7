import numpy as np
import cv2
import h5py
import time
import threading
import face_recognition
import os

class Artemis:

	def __init__(self, memory_path, cap_device = 0, skip_frames = 30):
		self.memory_path = memory_path
		self.skip_frames = skip_frames
		self.names = []
		self.faces = []
		self.last_seens = []
		
		with h5py.File(self.memory_path) as f:
			for name in f:
				self.names.append(name)
				self.faces.append(f[name]['face'][()])
				self.last_seens.append(f[name].attrs['last_seen'])

		self.cap = cv2.VideoCapture(cap_device)
		time.sleep(0.1)

		self.artemis_online = True

		if not os.path.isdir('UnknownFaces'):
			os.mkdir('UnknownFaces')

		self.main_thread = threading.Thread(target = self.main_loop)
		self.main_thread.start()

	def __del__(self):
		self.artemis_online = False
		self.main_thread.join()
		self.cap.release()

	def recognize_people(self,frame):
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		boxes = face_recognition.face_locations(rgb, model='cnn')
		encodings = face_recognition.face_encodings(rgb, boxes)
		
		detected_faces = []
		threshold = 0.4

		print(len(encodings))
		
		for i in range(len(encodings)):
			norms = np.linalg.norm(encodings[i] - np.asarray(self.faces), axis = 1)
			print(norms)
			for face in detected_faces:
				norms[self.names.index(face)] = 10
			ind = np.argmin(norms)
			
			if norms[ind] < threshold:
				detected_faces.append(self.names[ind])
				
				# Update this with weighted sum based on distance
				self.faces[ind] = 0.99*self.faces[ind] + 0.01*encodings[i]

				self.last_seens[ind] = time.time()

			else:
				(top, right, bottom, left) = boxes[i]
				top -= 0.05*frame.shape[0]
				bottom += 0.05*frame.shape[0]
				right += 0.05*frame.shape[1]
				left -= 0.05*frame.shape[1]
				top = int(top)
				bottom = int(bottom)
				left = int(left)
				right = int(right)
				if top < 0:
				    top = 0
				if bottom >= frame.shape[0]:
				    bottom = frame.shape[0]-1
				if left < 0:
				    left = 0
				if right >= frame.shape[1]:
				    right = frame.shape[1]-1

				cv2.imwrite('UnknownFaces/'+str(int(time.time()*100))+'.jpg', frame[top:bottom,left:right])	
				detected_faces.append('unk')


		self.people_in_room = list(detected_faces)

	def main_loop(self):
		frame_count = 0
		while self.artemis_online:
			ret, frame = self.cap.read()

			if ret is False:
				print('Artemis is blind!')

			if frame_count % self.skip_frames == 0:
				self.recognize_people(frame)
				print(self.people_in_room)

			frame_count += 1

art = Artemis('face_memory.hdf5')
time.sleep(10)
art.artemis_online = False