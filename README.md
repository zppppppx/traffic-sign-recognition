# traffic-sign-recognition

> 项目环境为树莓派系统，主要功能为使用摄像头捕捉实时画面处理得到交通标志以及其内容，从而控制小车的运动

<div align='center' ><font size='6'>小车路标识别</font></div>

[TOC]

[a = 1]

## A. Goal & Requirements

We need to recognize the traffic signs as follows [^1] :

![Alt text](https://i.loli.net/2021/07/15/3pscYMGqj8CAFvg.png "Traffic signs as examples")

## B. Steps & Designs

1. Collect enough traffic icons and label them to complete the production of the data set;
2. Select the features with good effect to extract the data set, and train the SVM vector machine;
3. Use OpenCV to complete the selection of the ROI area;
4. Complete SVM and template matching comprehensive recognition for the ROI area;
5. Complete the code of the movement part of the trolley, and control the movement of the trolley according to the recognition result.

## C. Environments

+ Raspberry Pi 4B+ Expansion Board for the movement of the trolley.
+ python >= 3.6
+ OpenCV >= 4.4
+ sklearn

## D. Main Steps

### D.1. Procession of the Datasets

In this section, we collect data sets and process them to obtain the traffic signs we need, and finally extract their HOG features and store them in a txt file for training the classifier.

#### D.1.1 Collection of the Datasets:

The data set can be searched on the Internet. In order to ensure the effect of subsequent classification, it is necessary to find as much data as possible. At the same time, the icons in the data set should be as clear as possible and contain various geometric states as much as possible, so as to ensure that the obtained classifier is high enough Robustness. The source of our data set is mainly Internet pictures and some open source projects (such as https://cg.cs.tsinghua.edu.cn/traffic-sign/).

![Alt text](https://i.loli.net/2021/07/15/voXLf76qJVPgchQ.png "Examples of the datasets")

#### D.1.2 The Extraction of ROI

The data used for classifier training is the image of the traffic icon, so the data set we collected cannot be used directly for the training of the classifier. The data set needs to be preprocessed first, and the traffic icons in the data set are framed, and then  intercept the traffic icon and put a label in its file name. The tool used to frame pictures is labelImg (project hosting address: https://github.com/tzutalin/labelImg). This tool can frame and label the traffic icons in the picture to form an xml file for subsequent pictures, providing convenience for later procession.

![Alt text](https://i.loli.net/2021/07/15/YPSwUCKIkRTWevl.png "Examples of xml files")

After completing the image tagging work, we perform image batch processing, read the path in the xml file, find the image in the original data set and use OpenCV to read it, and use the information in the ***bndbox*** to crop the image data to obtain us The desired target picture can be named and stored according to the category and number.

#### D.1.3 Feature Extraction for the Datasets

The training of the classifier needs to use the characteristics of the obtained traffic sign picture as the data input, so we need to extract the characteristics of the traffic sign. In this project we used HOG features.


+ The core idea of HOG is that the shape of the detected local object can be described by the light intensity gradient or the distribution of the edge direction. By dividing the entire image into small connected areas (called cells), each cell generates a directional gradient histogram or the edge direction of the pixel in the cell. The combination of these histograms can represent the description of the (target of the detected target) child. In order to improve the accuracy, the local histogram can be compared and standardized by calculating the light intensity of a larger area (called a block) in the image as a measure, and then using this value (measure) to normalize all cells in the block. This normalization One process completes better illumination/shadow invariance. 

+ Compared with other descriptors, the descriptors obtained by HOG maintain the invariance of geometric and optical transformations (unless the orientation of the object changes). Therefore, the HOG descriptor is particularly suitable for human detection. The same traffic signs also have a large degree of invariance, so here we believe that HOG features can robustly describe the characteristics of traffic signs, so as to obtain a good classification effect.

We use the scikit-image library for feature extraction. Each time we read a picture of the traffic sign obtained in **The Extraction of ROI**, use skimage to read its characteristics, and store its characteristics and signs in a txt file.

![Alt text](https://i.loli.net/2021/07/15/4uiASLXI5jpW1vq.png "Examples of features")

### D.2 Classification with OpenCV and SVM

#### D.2.1 The selection of the Region of the Sign

There are many ways to select feature regions, such as using gradient operations or morphological methods to obtain the boundary of the region, using gray image threshold segmentation to select the region of interest (ROI-Region of Interest). All in all, we can choose different methods to process images for different tasks, so as to achieve good segmentation performance.

In this project, the background changes during the recognition process are very complicated, and it is difficult to obtain good segmentation performance using segmentation methods such as grayscale images. We noticed that traffic signs have very distinctive features, such as most red or blue areas; or a complete red border as the boundary, so we can consider processing the image in the color space, so as to segment us Required area. In this project, we map the original RGB domain to the HSV space, and extract the color components in the HSV space.

+ HSV is a color space created by A. R. Smith in 1978 based on the intuitive characteristics of colors, also known as the Hexcone Model. Where H is hue, S is saturation, and V is lightness. HSV space is a specific form of HSI space, which is a hexagonal pyramid. Taking blue as an example, we generally think that the color with H in [100, 124], S in [43, 255], and V in [46, 255] is blue.

We use OpenCV to construct a mask, extract and binarize the color regions that meet the conditions (blue and red) to obtain a binary feature region representation, and then extract the boundary of the binarized region for frame selection, you can Get ROI.

#### D.2.2 The classification of the ROI

Next, we need to classify the ROI. In the field of image processing, the methods generally used to identify pictures include pattern matching, feature vector matching, and neural networks. In some basic problems, we can use pattern matching to obtain a good effect, but when the templates are relatively similar or there are a large number of categories that need to be identified, pattern matching may take a lot of time and may turn out not so accurate, such as the following two categories:

![Alt text](https://i.loli.net/2021/07/15/aRYWNT8yl7hdAEs.png "The road for right and Turn right")

The arrows in the center area have very similar features. If only template matching is used, if there is a partial occlusion in the picture, it is likely to be classified incorrectly. Therefore, we used another classification method in this project: SVM classification based on image features.



+ Support vector machines (SVM) is a binary classification model. Its basic model is a linear classifier with the largest interval defined in the feature space. The largest interval makes it different from the perceptron; SVM also includes kernel techniques , Which makes it an essentially non-linear classifier. The learning strategy of SVM is to maximize the interval, which can be formalized as a problem of solving convex quadratic programming, which is also equivalent to the problem of minimizing the regularized hinge loss function. The learning algorithm of SVM is the optimal algorithm for solving convex quadratic programming

We use the txt file obtained in the data processing part to train the SVM. The first idea we use is to directly classify all categories and directly obtain the end-to-end SVM, but after the actual training, we found that the SVM obtained by such training is for similar pictures It still does not have a good classification effect. We believe that it is because the feature vectors of these categories are also very similar. When the number of classification categories is large (27 in this project), these features cannot be completely linearly separable. The boundaries are very small.

We adjusted the methods of classification later and carried out hierarchical SVM classification. First, we map the original 27 categories to 7 categories, namely Speed, Stop, Direction, Road, Restrict, Yield, Background, and train the first-level SVM, and then train the second-level SVM under the Direction, Road, and Restrict categories. This strategy greatly improves the accuracy of recognition.

### D.3 Classification with Pattern Matching

For simplicity, we didn't finish all the classification work with SVM, since the speed class covers so many sub-classes,  so we turn to pattern matching after we recognize the speed sign.

#### D.3.1 The Basic Theory of Pattern Matching

Template matching is to use several known templates to match the template to be identified. When matching, you can compare the similarity of the features of the two images based on the various features of the two images, such as pixel distribution, contour, and some other features. You can use the Euclidean distance between the two data and the correlation coefficient, Cosine distance, etc. After the picture to be matched is matched with the template picture to find the similarity, the picture with the greatest correlation is found, which is the recognition result.

Obviously, this method has certain limitations. If the image to be recognized is rotated or occluded compared to the template, the similarity may decrease. In order to solve this problem, on the one hand, you can find a way to find some non-rotating images. The feature of denaturation, on the other hand, can add a template rotated by a certain angle to the matching template library to improve the robustness of recognition.

#### D.3.2 Making the pattern [^2]

In order to have a good matching effect, the digital template should occupy the largest area of the picture as much as possible. After the digital picture is collected, the picture can be accurately intercepted by the screenshot tool.

Since the numbers are all black, but the pictures obtained by intercepting or taking pictures will have certain color blocks of other colors, or the black is not pure enough, you can use some image processing methods to process the binary pure digital pictures, see the processing code The main idea of the matlab code in the attachment is to **blur the picture in order to filter out the noise**, and then perform the grayscale, and the binarization process, set a threshold, the gray value higher than this value will become black, otherwise It is white, and finally the color is reversed to obtain a black and white icon for easy comparison. Through this method, a clearer binary digital template that meets the requirements can be obtained.

![Alt text](https://i.loli.net/2021/07/15/zo3JNkZxILgit5r.png "Original pictures and patterns obtained")

#### D.3.3 Procession of Pattern Matching

Template matching is used after the speed limit icon has been detected by the support vector machine method and the specific position of the icon has been selected by the box. Therefore, this preprocessing mainly completes the detection, cutting, and matching pre-processing steps of the number in the digital icon.

The main idea of this step is to perform grayscale processing and binarization of the picture first, and then connect some broken and connected places through morphological closing operations to ensure the integrity of the digital contour, and finally find all of them through the OpenCV built-in function ***findContours()***. Then we select the circumscribed rectangular box of the outline, and perform template matching on the picture in the rectangular box.

![Alt text](https://i.loli.net/2021/07/15/7AgcLMK9wlTYumr.png "Original pictures & binarized pictures & segmented results")

After the binary processing and cutting the number, you can directly match it with the template stored in the specified path, use OpenCV's built-in function ***matchTemplate ()*** to find the similarity, and add the similarity to a list, Finally, find the maximum subscript in the list, use it to get the template file name, and finally get the matching result.

### D.4 The Control of Trolley and Multi-threads[^3]

#### D.4.1 Basic Structure of Trolley

The car used in the experiment is driven by a DC motor, and the large current required to drive the DC motor cannot be provided by the GPIO port of the Raspberry Pi. An external driver board is required, which is controlled by a chip such as L298N. Each motor needs to be controlled by positive and negative signals. In order to control the speed of the vehicle, it can also be controlled by a PWM pulse width modulation signal. 

In order to facilitate the call of other functions, the code that controls the movement of the car is encapsulated into a class. The actual process is to initialize the GPIO port that needs to be controlled to make it an output port, and then set its PWM frequency to 200Hz, and then turn right after going forward and backward. Set the forward and reverse rotations of different motors in the function to cooperate with the control of the direction of movement of the trolley. See sudu.py for the code. It should be noted that the knowledge of these four functions controls the direction of rotation of the motor. As for the result of the movement of the trolley, it should be determined by time. Therefore, for the left turn and the right turn, the actual implementation needs to be After the left turn and right turn function is executed for a period of time, the car can achieve the effect of turning left and right.

#### D.4.2 Why Multi-threads & Realization

In the process of running the whole program, it can be simply considered that there are two tasks to be executed at the same time, one is the movement of the car, and the other is the recognition of the image captured by the camera of the car. The car still needs to keep moving during the recognition process. , So it is necessary to turn on dual threads to achieve the purpose of simultaneous movement of the trolley and image recognition.

Multithreading means that there can be two or more threads in a program, or two tasks can run at the same time, and data can be shared between the two threads. In python, you can use the threading class to perform multi-threaded tasks. In addition to the main thread running, define an ***autoThread*** class. There is an in-class function run() for tasks that require multi-threaded operation. In the code, the image recognition is set to the main thread, and the sub-thread is set to run the car. When the main thread recognizes the traffic icon and returns the result, it changes the value of a global variable, and then calls the global variable in the sub-thread because The change of global variables will respond accordingly and adjust the motion state. The specific effect can be seen in the demonstration video.

## E. Summary

So far, we have realized the function of recognizing the traffic icons in the camera through the Raspberry Pi, and then controlling the speed and direction of the movement of the car.

Thanks for reading and your kind opinions!







[^1]: The specific signs can be decided by the users, with corresponding datasets.
[^2]: To assure robustness, we can add multiple patterns including oblique numbers.
[^3]: You can find plenty of tutorials on the Internet, so we just add some brief introductions in this part.
