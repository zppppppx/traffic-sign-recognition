import os
import numpy as np 
import cv2

def print_res(res):
	speed = 30
	if res!=None:
		if 8 in res :
			speed = 80
		elif 3 in res:
			speed = 30
		elif 1 in res:
			speed = 10
		else:
			speed = 30
	return speed
#######
def traffic_detection(src):
	col,row = src.shape[:2]
	img=src
	img=cv2.resize(img,(500,460))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
	ker = np.ones((6, 6), np.uint8)
	close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ker)
	h, w = gray.shape[0], gray.shape[1]
	point1 = [0.12 * w, 4*h /16]
	point2 = [0.12 * w, 12 * h /16]
	point3 = [0.6 * w, 12 * h / 16]
	point4 = [0.6 * w, 4*h / 16]
	list1 = np.array([[point1, point2, point3, point4]], dtype=np.int32)
	mask = np.zeros_like(gray)
	mask = cv2.fillConvexPoly(mask, list1, 255)
	mask1 = cv2.bitwise_and(mask, thresh)

	ker = np.ones((6, 6), np.uint8)
	mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, ker)
	ker = np.ones((3, 3), np.uint8)
	mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, ker)
	# cv2.imshow('img',mask1)
	contours1, hierarchy1 = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	a = len(contours1)

	list2 = []
	list3=[]
	res=[]
	for lk in contours1:
		x1, y1, w1, h1 = cv2.boundingRect(lk)
		list2.append([x1,y1,w1,h1])
		list3.append(w1*h1)
	x1,y1,w1,h1=list2[list3.index(max(list3))]
	roi = mask1[y1:y1 + h1, x1:x1 + w1]
	roi = cv2.resize(roi, (60, 90))
	roi=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
	filename = os.listdir('./template')
	# print(filename)
	scores = []
	for i in range(len(filename)):
		src1 = cv2.imread('template/' + filename[i])
		src1 = cv2.resize(src1, (60, 90))
		result = cv2.matchTemplate(src1, roi, cv2.TM_CCOEFF_NORMED)
		# cv2.imshow('roi',roi)
		(_, score, _, _) = cv2.minMaxLoc(result)
		scores.append(score)

	x3 = np.argmax(scores)
	name = filename[x3]
	res.append(int(name[0]))
	speed = print_res(res)
	return speed
	return 0