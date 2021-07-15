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
    mapping = {'0':0, '3':1, '1':2, '2':2, '4':2, '5':2, '6':2, '7':2, '8':2, '9':3, '10':3, '11':3, '12':3,
                '13':3, '14':3, '25':3, '15':4, '16':4, '17':4, '18':4, '19':4, '20':4, '21':4, '22':4, 
                '23':5, '24':6}
    clf = SVC(C=10, tol=1e-3, probability = True, gamma='auto')
    main_labels = [*map(lambda x: mapping[x], labels)]
    clf.fit(hog_features, main_labels)
    joblib.dump(clf, save_path)
    print('Main classes training is finished')

def sub_cut(labels, sub_dicts):
    keys = list(sub_dicts.keys())
    indexes = []
    for i in range(len(labels)):
        if labels[i] in keys:
            indexes.append(i)
    return indexes

def Sub_classes_train(hog_features, labels, sub_dict, save_path='.\Sub_classes.pkl'):
    indexes = sub_cut(labels, sub_dict)
    labels = labels[indexes]
    hog_features = hog_features[indexes]
    sub_labels = [*map(lambda x: sub_dict[x], labels)]
    clf = SVC(C=10, tol=1e-3, probability = True, gamma='auto')
    clf.fit(hog_features, sub_labels)
    joblib.dump(clf, save_path)
    cls = save_path.split('\\')[-1]
    cls = cls.split('.')[0]
    print('Subclasses '+cls +' finished')



if __name__ == '__main__':
    hog_train_txt = "E:\projects\\recognition\online\Traffic_sign_recognition-master\data\hog.txt"
    Main_classes = "E:\projects\\recognition\online\Traffic_sign_recognition-master\Main_classes.pkl"
    Sub_classes = ['',
                    '',
                    'E:\projects\\recognition\online\Traffic_sign_recognition-master\Direction.pkl',
                    'E:\projects\\recognition\online\Traffic_sign_recognition-master\Road.pkl',
                    'E:\projects\\recognition\online\Traffic_sign_recognition-master\Restrict.pkl',
                    '',
                    '']

        
    img_names, labels, hog_train_features = load_hog_data(hog_train_txt)
    print(set(labels))
    Main_classes_train(hog_train_features, labels, Main_classes)
    for i in range(7):
        sub_class = Sub_classes[i]
        if sub_class == '':
            continue
        Sub_classes_train(hog_train_features, labels, Main_Classes[i], save_path=sub_class)
