import numpy as np 
import os
from skimage import feature as ft 
import cv2

img_label = {"Speed": 0, "Background": 1}
img_label = {"Speed":0, "Right": 1, "Left": 2, "Circle": 3, "Pedestrain_restrict": 4, "Background": 5}
img_label = {"Speed": 0, "Right":1, "Background": 2}
img_label = {"Speed":0, "Right": 1,'Left':2 ,'Stop':3,'Straight':4,'Strai_Right':5,'Strai_Left':6,
			'Left_Right':7,'Circle':8,'Main_Road':9,'Pedestrain':10,'Turn_around':11,'Road_right':12,
			'Road_strai':13,'Road_strai_right':14,'Right_restrict':15,
			'Left_restrict':16,'Straight_restrict':17,'Left_Right_restrict':18,'Van_restrict':19,
			'Long_parking_restrict':20,'Honk_restrict':21,'Pedestrain_restrict':22,'Yield':23,'Background':24, 
			'Pedestrain_B':25}

# ################
# # 已有的类别	#
# ################
# img_label = {"Speed":0, "Right": 1,'Left':2 ,'Straight':3,'Strai_Right':4,'Strai_Left':5,
# 			'Left_Right':6,'Circle':7,'Pedestrain':8,'Turn_around':9,
# 			'Right_restrict':10,
# 			'Left_restrict':11,'Straight_restrict':12,'Left_Right_restrict':13,
# 			'Honk_restrict':14,'Yield':15,'Background':16}

            
# #############
# # 视频类别	#
# ############
# img_label = {'Speed':0, 'Right':1,'Left':2,'Background':3}


def hog_feature(img_array, resize=(64,64)):
    ##提取HOG特征

    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, resize)
    bins = 9
    cell_size = (8, 8)
    cpb = (2, 2)
    norm = "L2"
    features = ft.hog(img, orientations=bins, pixels_per_cell=cell_size, 
                        cells_per_block=cpb, block_norm=norm, transform_sqrt=True)
    return features

def extra_hog_features_dir(img_dir, write_txt, resize=(64,64)):
    ##提取目录中所有图像HOG特征
   
    img_names = os.listdir(img_dir)
    img_names = [os.path.join(img_dir, img_name) for img_name in img_names]
    if os.path.exists(write_txt):
        os.remove(write_txt)
    
    with open(write_txt, "a") as f:
        index = 0
        for img_name in img_names:
            img_array = cv2.imread(img_name)
            features = hog_feature(img_array, resize)
            label_name = img_name.split("\\")[-1].split("__")[0]
            label_num = img_label[label_name]
            row_data = img_name + "\t" + str(label_num) + "\t"
            
            for element in features:
                row_data = row_data + str(round(element,3)) + " "
            row_data = row_data + "\n"
            f.write(row_data)
            
            if index%100 == 0:
                print ("total image number = ", len(img_names), "current image number = ", index)
            index += 1


if __name__ == "__main__":
    img_dir = "E:\projects\\recognition\online\Traffic_sign_recognition-master\data\\realTrain\datatrain"
    write_txt = "E:\projects\\recognition\online\Traffic_sign_recognition-master\data\\hog.txt"
    extra_hog_features_dir(img_dir, write_txt, resize=(64,64))
    print ("done")
