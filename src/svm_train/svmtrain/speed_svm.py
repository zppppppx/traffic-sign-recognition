import numpy as np 
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


classes_num = {"Speed":0, "Right": 1,'Left':2 ,'Stop':3,'Straight':4,'Strai_Right':5,'Strai_Left':6,
			'Left_Right':7,'Circle':8,'Main_Road':9,'Pedestrain':10,'Turn_around':11,'Road_right':12,
			'Road_strai':13,'Road_strai_right':14,'Right_restrict':15,
			'Left_restrict':16,'Straight_restrict':17,'Left_Right_restrict':18,'Van_restrict':19,
			'Long_parking_restrict':20,'Honk_restrict':21,'Pedestrain_restrict':22,'Yield':23,'Background':24, 
			'Pedestrain_B':25}

Speed = {'0':0}

Stop = {'3':0}

Direction = {'1':0, 
            '2':1, 
            '4':2, 
            '5':3, 
            '6':4,
            '7':5,
            '8':6}

Road = {'9':0,
        '10':1,
        '11':2,
        '12':3,
        '13':4,
        '14':5,
        '25':6}

Restrict = {'15':0,
			'16':1,
			'17':2,
			'18':3,
			'19':4,
			'20':5,
			'21':6,
			'22':7}

Yield = {'23':0}

Background = {'24':0}

Main_Classes = [Speed, Stop, Direction, Road, Restrict, Yield, Background]

mapping = {'Speed':'Speed', 
			'Right':'Direction',
			'Left':'Direction',
			'Stop':'Stop',
			'Straight':'Direction',
			'Strai_Right':'Direction',
			'Strai_Left':'Direction',
			'Left_Right':'Direction',
			'Circle':'Circle',
			'Main_Road':'Main_Road',
			'Pedestrain':'Pedestrain',
			'Turn_around':'Direction',
			'Road_right':'Direction',
			'Road_strai':'Direction',
			'Road_left':'Direction',
			'Road_strai_right':'Direction',
			'Road_strai_left':'Direction',
			'Right_restict':'Restrict',
			'Left_restrict':'Restrict',
			'Straight_restrict':'Restrict',
			'Left_Right_restrict':'Restrict',
			'Van_restrict':'Restrict',
			'Long_parking_restrict':'Restrict',
			'Honk_restrict':'Restrict',
			'Pedestrain_restict':'Restrict',
			'Yield':'Yield',
			'Background':'Background',
			'Pedestrain_B':'Pedestrain'}


def load_hog_data(hog_txt):

    img_names = []
    labels = []
    hog_features = []
    with open(hog_txt, "r") as f:
        data = f.readlines()
        for row_data in data:
            row_data = row_data.rstrip()
            img_path, label, hog_str = row_data.split("\t")
            img_name = img_path.split("/")[-1]
            hog_feature = hog_str.split(" ")
            hog_feature = [float(hog) for hog in hog_feature]
            #print "hog feature length = ", len(hog_feature)
            img_names.append(img_name)
            labels.append(label)
            hog_features.append(hog_feature)
    return img_names, np.array(labels), np.array(hog_features)

def Main_classes_train(hog_features, labels, save_path='.\Main_classes.pkl'):
    clf = SVC(C=10, tol=1e-3, probability = True, gamma='auto')

    clf.fit(hog_features,labels)
    joblib.dump(clf, save_path)
    print('Main classes training is finished')


if __name__ == '__main__':
    hog_train_txt = r".\hog.txt"
    Main_classes = r'.\Speed.pkl'

        
    img_names, labels, hog_train_features = load_hog_data(hog_train_txt)
    print(set(labels))
    Main_classes_train(hog_train_features, labels, Main_classes)
