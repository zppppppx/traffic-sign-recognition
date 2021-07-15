import os
import numpy as np 
import cv2
from skimage import feature as ft 
from sklearn.externals import joblib
from template_recognize import traffic_detection
##############
# Clf files ##
##############
Direction_clf = joblib.load("./pkls/Direction.pkl")
Road_clf = joblib.load("./pkls/Road.pkl")
Restrict_clf = joblib.load("./pkls/Restrict.pkl")
Clfs = [None, None, Direction_clf, Road_clf, Restrict_clf, None, None]

##############
# Sub dicts ##
##############
Speed = ['Speed']
Stop = ['Stop']
Direction = ['Right', 'Left', 'Straight', 'Strai_Right', 'Strai_Left', 'Left_Right', 'Circle']
Road = ['Main_Road', 'Pedestrain', 'Turn_around', 'Road_right', 'Road_strai', 'Road_strai_right', 'Pedestrain_B']
Restrict = ['Right_restrict', 'Left_restrict', 'Straight_restrict', 'Left_Right_restrict', 'Van_restrict', 
			'Long_parking_restrict', 'Honk_restrict', 'Pedestrain_restrict']
Yield = ['Yield']
Background = ['Background']
sub_dicts = [Speed, Stop, Direction, Road, Restrict, Yield, Background]


#################
# Main_classes	#
#################
cls_names = ['Speed', 'Stop', 'Direction','Road', 'Restrict', 'Yield', 'Background']
img_label = {'Speed':0, 'Stop':1, 'Direction':2, 'Road':3, 'Restrict':4, 'Yield':5, 'Background':6}

# cls_names = ['Right_restrict', 'Left_restrict', 'Straight_restrict','Left_Right_restrict', 
#             'Van_restrict', 'Long_parking_restrict', 'Honk_restrict','Pedestrain_restrict']
# img_label = {'Right_restrict':0, 'Left_restrict':1, 'Straight_restrict':2,'Left_Right_restrict':3, 
#             'Van_restrict':4, 'Long_parking_restrict':5, 'Honk_restrict':6,'Pedestrain_restrict':7}

'''

'''

def preprocess_img(imgBGR):

	imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
	Bmin = np.array([110, 43, 46])
	Bmax = np.array([124, 255, 255])
	Bmin = np.array([110, 150, 46])
	Bmax = np.array([124, 255, 255])

	img_Bbin = cv2.inRange(imgHSV,Bmin, Bmax)
	Rmin2 = np.array([165, 43, 46])
	Rmax2 = np.array([180, 255, 255])
	Rmin2 = np.array([165, 43, 46])
	Rmax2 = np.array([180, 255, 255])
	img_Rbin2 = cv2.inRange(imgHSV,Rmin2, Rmax2)
	Rmin1 = np.array([0,43,46])
	Rmax1=np.array([10,255,255])
	img_Rbin1 = cv2.inRange(imgHSV,Rmin1, Rmax1)
	img_bin = np.maximum(img_Bbin, img_Rbin2,img_Rbin1)
	# cv2.imshow('cam',img_bin)
	return img_bin

'''

'''
def contour_detect(img_bin, min_area=0, max_area=-1, wh_ratio=2.0):
	rects = []

	contours, hierarchy= cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if len(contours) == 0:
		return rects
	max_area = img_bin.shape[0]*img_bin.shape[1] if max_area<0 else max_area
	for contour in contours:
		area = cv2.contourArea(contour)
		if area >= min_area and area <= max_area:
			x, y, w, h = cv2.boundingRect(contour)
			if 1.0*w/h < wh_ratio and 1.0*h/w < wh_ratio:
				rects.append([x,y,w,h])
	return rects
'''

'''
def draw_rects_on_img(img, rects):
	img_copy = img.copy()
	for rect in rects:
		x, y, w, h = rect
		cv2.rectangle(img_copy, (x,y), (x+w,y+h), (0,255,0), 2)
	return img_copy

def hog_extra_and_svm_class(proposal, clf, resize = (64, 64)):

	img = cv2.cvtColor(proposal, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, resize)
	bins = 9
	cell_size = (8, 8)
	cpb = (2, 2)
	norm = "L2"
	features = ft.hog(img, orientations=bins, pixels_per_cell=cell_size, 
		cells_per_block=cpb, block_norm=norm, transform_sqrt=True)
	features = np.reshape(features, (1,-1))
	cls_prop = clf.predict_proba(features)
	cls_prop = cls_prop[0]
	return cls_prop
'''

'''
def Sub_division(proposal, clf, sub_cls):
	cls_prop = hog_extra_and_svm_class(proposal, clf)
	cls_prop = np.round(cls_prop, 2)
	cls_num = np.argmax(cls_prop)
	result = sub_cls[cls_num]
	return result
'''
'''
def Sort_command(proposal, clf,src):
	cls_prop = hog_extra_and_svm_class(proposal, clf)
	cls_prop = np.round(cls_prop, 2)
	cls_num = np.argmax(cls_prop)
	if cls_num == 0:
		"""speed
		"""
		speed = traffic_detection(proposal)
		return speed
	if cls_num == 1:
		"""Here we need to control the car to stop"""
		return 'Stop'
	if cls_num in range(2,5):
		sub_cls = sub_dicts[cls_num]
		result = Sub_division(proposal, Clfs[cls_num], sub_cls)
		return result
	else:
		result = cls_names[cls_num]
		return result
	
