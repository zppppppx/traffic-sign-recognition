import os
import numpy as np 
import cv2
from skimage import feature as ft 
from sklearn.externals import joblib
import time
import threading
import ctypes

from hog_recognize import *
from template_recognize import traffic_detection

def _async_raise(tid, exctype):
	tid = ctypes.c_long(tid)
	if not inspect.isclass(exctype):
		exctype = type(exctype)
	res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
	if res == 0:
		raise ValueError("invalid thread id")
	elif res != 1:
		ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
		raise SystemError("PyThreadState_SetAsyncExc failed")
	print('stop automode')
def stop_thread(thread):
	_async_raise(thread.ident, SystemExit)


class autoThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
	def run(self):
		if motion_para == 10:
			print('speed',10)
		elif motion_para == 30:
			print('speed',30)
		elif motion_para == 80:
			print('speed',80)
		elif motion_para == 'Right':
			time.sleep(0.8)
			print('speed right')
		elif motion_para == 'Straight':
			print('speed straight')
		elif motion_para == 'Left':
			time.sleep(0.8)
			print('speed left')
		elif motion_para == 'Stop':
			print('speed stop')
motion_para = 30
speed = 80
if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	# cap = cv2.VideoCapture('./car.mp4')
	# cap = cv2.VideoCapture('./xiaoche.mp4')
	cv2.namedWindow('camera')
	cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	clf = joblib.load("./pkls/Main_classes.pkl")

	i=0
	##########
	A=autoThread()
	A.start()
	auto = True
	#######
	while (1):
		i+=1
		ret, img = cap.read()
		# ret = 1
		if ret:
			if i%3 == 0:
				img_bin = preprocess_img(img)
				min_area = img_bin.shape[0]*img.shape[1]/(25*25)
				rects = contour_detect(img_bin, min_area=min_area)
				if rects:
					Max_X=0
					Max_Y=0
					Max_W=0
					Max_H=0
					for r in rects:
						if r[2]*r[3]>=Max_W*Max_H:
							Max_X,Max_Y,Max_W,Max_H=r
					proposal = img[Max_Y:(Max_Y+Max_H),Max_X:(Max_X+Max_W)]
					cv2.rectangle(img,(Max_X,Max_Y), (Max_X+Max_W,Max_Y+Max_H), (0,255,0), 2)
					
					motion_para = Sort_command(proposal, clf,img)
					print(motion_para)
					A=autoThread()
					A.start()
					auto = True

				cv2.imshow('camera',img)
				cv2.waitKey(40)
		else:
			cv2.destroyAllWindows()
			cap.release()
			break
