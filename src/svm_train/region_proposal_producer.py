import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET


cls_names = ['Speed', 'Right','Left','Stop','Straight','Strai_Right','Strai_Left',
			'Left_Right','Circle','Combine','Pedestrain','Turn_around','Road_right',
			'Road_strai','Road_left','Road_strai_right','Road_strai_left','Right_restrict',
			'Left_restrict','Straight_restrict','Left_Right_restrict','Van_restrict',
			'Long_parking_restrict','Honk_restrict','Pedestrain_restict','Yield','Background','Pedestrain_B']

img_label = {'000':'Speed', 
			'001':'Right',
			'002':'Left',
			'003':'Stop',
			'004':'Straight',
			'005':'Strai_Right',
			'006':'Strai_Left',
			'007':'Left_Right',
			'008':'Circle',
			'009':'Main_Road',
			'010':'Pedestrain',
			'011':'Turn_around',
			'012':'Road_right',
			'013':'Road_strai',
			'014':'Road_strai_right',
			'015':'Right_restict',
			'016':'Left_restrict',
			'017':'Straight_restrict',
			'018':'Left_Right_restrict',
			'019':'Van_restrict',
			'020':'Long_parking_restrict',
			'021':'Honk_restrict',
			'022':'Pedestrain_restict',
			'023':'Yield',
			'024':'Background',
			'025':'Pedestrain_B'}

# target_class=["000","001","002","008","024"]##所要识别的交通指示牌
# classes_name = ["Speed","Right", "Left", "Circle", "Pedestrain_restrict", "Background"]
# classes_num = {"Speed":0, "Right": 1, "Left": 2, "Circle": 3, "Pedestrain_restrict": 4, "Background": 5}

# classes_name = ["Speed","Right", "Background"]
classes_name = ['Speed', 'Right','Left','Stop','Straight','Strai_Right','Strai_Left',
			'Left_Right','Circle','Main_Road','Pedestrain','Turn_around','Road_right',
			'Road_strai','Road_strai_right','Right_restrict',
			'Left_restrict','Straight_restrict','Left_Right_restrict','Van_restrict',
			'Long_parking_restrict','Honk_restrict','Pedestrain_restrict','Yield','Background', 'Pedestrain_B']

classes_num = {"Speed":0, "Right": 1,'Left':2 ,'Stop':3,'Straight':4,'Strai_Right':5,'Strai_Left':6,
			'Left_Right':7,'Circle':8,'Main_Road':9,'Pedestrain':10,'Turn_around':11,'Road_right':12,
			'Road_strai':13,'Road_strai_right':14,'Right_restrict':15,
			'Left_restrict':16,'Straight_restrict':17,'Left_Right_restrict':18,'Van_restrict':19,
			'Long_parking_restrict':20,'Honk_restrict':21,'Pedestrain_restrict':22,'Yield':23,'Background':24, 
			'Pedestrain_B':25}

# ################
# # 已有的类别	#
# ################
# classes_name = ['Speed', 'Right','Left','Straight','Strai_Right','Strai_Left',
# 			'Left_Right','Circle','Pedestrain','Turn_around',
# 			'Right_restrict',
# 			'Left_restrict','Straight_restrict','Left_Right_restrict',
# 			'Honk_restrict','Yield','Background']

# classes_num = {"Speed":0, "Right": 1,'Left':2 ,'Straight':3,'Strai_Right':4,'Strai_Left':5,
# 			'Left_Right':6,'Circle':7,'Pedestrain':8,'Turn_around':9,
# 			'Right_restrict':10,
# 			'Left_restrict':11,'Straight_restrict':12,'Left_Right_restrict':13,
# 			'Honk_restrict':14,'Yield':15,'Background':16}


# #############
# # 视频类别	#
# ############
# classes_name = ['Speed', 'Right','Left','Background']
# classes_num = {'Speed':0, 'Right':1,'Left':2,'Background':3}

SIGN_ROOT = "E:/projects/recognition/online/Traffic_sign_recognition-master"
DATA_PATH = os.path.join(SIGN_ROOT, 'data/realTrain/')


def parse_xml(xml_file):
	##返回该图片矩形对焦顶点坐标以及交通标志类型
	# print(xml_file)
	tree = ET.parse(xml_file)
	root = tree.getroot()
	image_path = ''
	labels = []

	for item in root:
		if item.tag == 'filename':
			image_path = os.path.join(DATA_PATH, "part1/", item.text)
		elif item.tag == 'object':
			obj_name = item[0].text
			# print(obj_name)
			if obj_name in classes_name:
				obj_num = classes_num[obj_name]
			else:
				obj_num = classes_num['Background']
			xmin = int(item[4][0].text)
			ymin = int(item[4][1].text)
			xmax = int(item[4][2].text)
			ymax = int(item[4][3].text)
			labels.append([xmin, ymin, xmax, ymax, obj_num])
	return image_path,labels

def produce_neg_proposals(img_path, write_dir, min_size, square=False, proposal_num=0):
	##返回其他类型的交通标志裁剪图的数目
	print(img_path)
	img = cv2.imread(img_path)
	rows = img.shape[0]
	cols = img.shape[1]
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	imgBinBlue = cv2.inRange(imgHSV,np.array([110,43,46]), np.array([124,255,255]))
	imgBinRed = cv2.inRange(imgHSV,np.array([165,43,46]), np.array([180,255,255]))
	imgBin = np.maximum(imgBinRed, imgBinBlue)

	contours, _ = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for contour in contours:
		x,y,w,h = cv2.boundingRect(contour)
		if w<min_size or h<min_size:
			continue

		if square is True:
			xcenter = int(x+w/2)
			ycenter = int(y+h/2)
			size = max(w,h)
			xmin = int(round(max(xcenter-size/2, 0),0))
			xmax = int(round(min(xcenter+size/2,cols),0))
			ymin = int(round(max(ycenter-size/2, 0),0))
			ymax = int(round(min(ycenter+size/2,rows),0))
			proposal = img[ymin:ymax, xmin:xmax]
			proposal = cv2.resize(proposal, (size,size))

		else:
			proposal = img[y:y+h, x:x+w]
		write_name = "Background" + "__" + str(proposal_num) + ".png"
		proposal_num += 1
		cv2.imwrite(os.path.join(write_dir,write_name), proposal)##保存其他类型交通标志裁剪后的图像
	return proposal_num

def produce_pos_proposals(img_path, write_dir, labels, min_size, square=False, proposal_num=0, ):
	##更新目标交通标志裁剪图的数目,返回proposal_num对象
	img = cv2.imread(img_path)
	rows = img.shape[0]
	cols = img.shape[1]
	for label in labels:
		# print(label)
		xmin, ymin, xmax, ymax, cls_num = np.int32(label)
		if xmax-xmin<min_size or ymax-ymin<min_size:
			continue
		if square is True:
			xcenter = int((xmin + xmax)/2)
			ycenter = int((ymin + ymax)/2)
			size = max(xmax-xmin, ymax-ymin)
			xmin = int(round(max(xcenter-size/2, 0)))
			xmax = int(round(min(xcenter+size/2,cols)))
			ymin = int(round(max(ycenter-size/2, 0)))
			ymax = int(round(min(ycenter+size/2,rows)))
			proposal = img[ymin:ymax, xmin:xmax]
			# print(proposal)
			proposal = cv2.resize(proposal, (size,size))
		else:
			proposal = img[ymin:ymax, xmin:xmax]
			# print(proposal)
		cls_name = classes_name[cls_num]
		# print('class_name = ', cls_name)
		proposal_num[cls_name] +=1
		write_name = cls_name + "__" + str(proposal_num[cls_name]) + ".png"
		# print(os.path.join(write_dir,write_name))
		# print(proposal)
		# cv2.imwrite(os.path.join(write_dir,write_name), proposal)
		cv2.imwrite('E:\projects\\recognition\online\Traffic_sign_recognition-master\data\\realTrain\datatrain\\'+write_name, proposal)
	return proposal_num


def produce_proposals(xml_dir, write_dir, square=False, min_size=30):
                ##返回proposal_num对象
	proposal_num = {}
	for cls_name in classes_name:
		proposal_num[cls_name] = 0
               ##img_names = os.listdir(img_dir)
               ##img_names = [os.path.join(img_dir, img_name) for img_name in img_names]
	index = 0
	for xml_file in os.listdir(xml_dir):
		img_path, labels = parse_xml(os.path.join(xml_dir,xml_file))
		img = cv2.imread(img_path)
		# print('labels  =  ', labels)
		##如果图片中没有出现定义的那几种交通标志就把它当成负样本
		if len(labels) == 0:
			neg_proposal_num = produce_neg_proposals(img_path, write_dir, min_size, square, proposal_num["Background"])
			proposal_num["Background"] = neg_proposal_num
		else:
			proposal_num = produce_pos_proposals(img_path, write_dir, labels, min_size, square=True, proposal_num=proposal_num)
			
		if index%100 == 0:
			print ("total xml file number = ", len(os.listdir(xml_dir)), "current xml file number = ", index)
			print ("proposal num = ", proposal_num)
		index += 1

	return proposal_num

if __name__ == "__main__":
	xml_dir = "E:/projects/recognition/online/Traffic_sign_recognition-master/data/realTrain/Annotations"
	save_dir = "E:/projects/recognition/online/Traffic_sign_recognition-master/data/datatrain"
	proposal_num = produce_proposals(xml_dir, save_dir, square=True)
	print ("proposal num = ", proposal_num)

