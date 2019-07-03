import numpy as np
import cv2
import os

os.mkdir('face_data')

cap = cv2.VideoCapture(0)

for i in range(1000):
	ret, frame = cap.read()

	cv2.imwrite('face_data/'+str(i+1)+'.jpg',frame)

cap.release()	