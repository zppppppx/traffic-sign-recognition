import os
import numpy as np 
import cv2
from skimage import feature as ft 
from sklearn.externals import joblib
# from sudu import *
import time
import threading
import ctypes
# sudu=sudu()
####
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
		# print('fuck',motion_para)
		if motion_para == 10:
			speed = 60
			print('speed',10)
			# sudu.gogo(speed)
		elif motion_para == 30:
			speed = 80
			print('speed',30)
			# sudu.gogo(speed)
		elif motion_para == 80:
			speed = 100
			print('speed',80)
			# sudu.gogo(speed)
		elif motion_para == 'Right':
			# sudu.right(80)
			time.sleep(0.8)
			# sudu.gogo(80)
			# sudu.gogo(speed)
			print('speed right')
		elif motion_para == 'Straight':
			# sudu.gogo(speed)
			print('speed straight')
		elif motion_para == 'Left':
			# sudu.right(80)
			time.sleep(0.8)
			# sudu.gogo(80)
			# sudu.gogo(speed)
			print('speed left')
		elif motion_para == 'Stop':
			# sudu.stop()
			print('speed stop')
motion_para = 30
speed = 80
if __name__ == "__main__":
	clf = joblib.load("./pkls/Main_classes.pkl")
	filename = os.listdir('./data')
	cnt=0
	cnt_right=0
	for i in range(len(filename)):
		img = cv2.imread('data/' + filename[i])
		# img_bin = preprocess_img(img)
		# min_area = img_bin.shape[0]*img.shape[1]/(25*25)
		# rects = contour_detect(img_bin, min_area=min_area)
		# if rects:
		# 	Max_X=0
		# 	Max_Y=0
		# 	Max_W=0
		# 	Max_H=0
		# 	for r in rects:
		# 		if r[2]*r[3]>=Max_W*Max_H:
		# 			Max_X,Max_Y,Max_W,Max_H=r
		# 	proposal = img[Max_Y:(Max_Y+Max_H),Max_X:(Max_X+Max_W)]
		# 	cv2.rectangle(img,(Max_X,Max_Y), (Max_X+Max_W,Max_Y+Max_H), (0,255,0), 2)
		# 	# cv2.imshow("proposal", proposal)
			
		motion_para = traffic_detection(img)
		
		cnt+=1
		if str(motion_para) in filename[i]:
			cnt_right+=1
		# print(filename[i],motion_para)
		# print(cnt,cnt_right)
		# cv2.imshow('camera',img)
	print('The Correct rate is ',cnt_right/cnt)