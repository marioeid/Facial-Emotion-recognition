# Facial Emotion Recognition Project
## What Datasets we tried ?
- Fer , Fer+ , CK+ , EMOTIC 

## what General ideas we tried ?
- combining all the datasets into one dataset to get more data (didn't improve the accuracy that much )
- cutting out some of the output classes instead of 9 classes to predict let's try the top 4 or 5 classes with more data
- Cleaning the Data by hand , improved alot of the accuracy but we didn't have access to some amazing datasets like AffectNet
- Data agumentation worked and k-cross validation improved 2% to 4% on overall accuracy and worked like a charm .
- trying normal CNN and transfer learning
- trying media pipe keypoints as our inputs

## What Algorithms We Tried in the Start ?
- normal CNN - Transfer Learning (Vgg-Face , Vgg19 ,Vgg18 , ResNet50) and more
- we used Fer+ dataset and Extracted the Facials using Media pipe and feeded the extracted feautres into our neural network 

## what accurcies we got 
- normal CNN didn't work well on Fer highest we could get was 70 to 72 using k-cross validation 
- we got over 90% using transfer learning we got 3% more accuracy using K-cross validation and class weights .
- we got 97% using mediapipe feature extraction on Fer + dataset ( it was forked from a repo ) we wanted to try k-cross validation and data agumentation but we didn't have access to the dataset keypoints csv we hoped he uploaded that ( we think adding some techniques can improve the accuracy by at least 1 to 2 percent more )

## what problems we faced ?
- Face Detection in real time might really be not accurate 
- most of those algorithms were too fast on real time and didn't actually work that well
- many algorithms were biased to one emotion and it was due that emotion was the dominant emotion in the dataset 


## how we solved those problems ?
- we can use a sperate library like deep face for face detection then detecting the emotions or we can use facemesh to determine both the face and the keypoints
- deep face library has wide range of algorithms we recommend using opencv or mediapipe for best balance between speed and accuracy on real time , mediapipe 5 meters algorithms also works like a charm in real time
- Lstm models did pretty well on real time, rather than predicting on one frame we predict after multiple frames
- we can also predict one frame and make it display for multiple milli seconds to be clear to the eye 
- we used class weights and down sampling and up sampling techniques to improve the accuracy 
- we didn't face that problem with mediapipe 

## conclusion 
- extracting the features from mediapipe and feeding it to the neural network is the best approach by far but you need to be carefull with the data set
## 
# Reference from REWTAO 
- https://github.com/REWTAO/Facial-emotion-recognition-using-mediapipe
- we hope u add the keypoints csv

# Reference from Kazuhito Takahashi(https://twitter.com/KzhtTkhs)
- [MediaPipe](https://mediapipe.dev/)
- [Kazuhito00/mediapipe-python-sample](https://github.com/Kazuhito00/mediapipe-python-sample)
- [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)

# facial emotion recognition using mediapipe
- Estimate face mesh using MediaPipe(Python version).This is a sample program that recognizes facial emotion with a simple multilayer perceptron using the detected key points that returned from mediapipe.Although this model is 97% accurate, there is no generalization due to too little training data.
- the project is implement from https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe to use in facial emotion recognition
- the keypoint.csv is empty because this file is too large to upload so if you want to training model please find new dataset or record data by yourself

This repository contains the following contents.
- Sample program
- Facial emotion recognition model(TFLite)
- Script for collect data from images dataset and camera 

# Requirements
- mediapipe 0.8.9
- OpenCV 4.5.4 or Later
- Tensorflow 2.7.0 or Later
- scikit-learn 1.0.1 or Later (Only if you want to display the confusion matrix) 
- matplotlib 3.5.0 or Later (Only if you want to display the confusion matrix)

### main.py
This is a sample program for inference.it will use keypoint_classifier.tflite as model to predict your emotion.

### training.ipynb
This is a model training script for facial emotion recognition.

### model/keypoint_classifier
This directory stores files related to facial emotion recognition.
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### Collect_from_image.py
This script will collect the keypoints from image dataset(.jpg). you can change your dataset directory to collect data.It will use your folder name to label.

### Collect_from_webcam.py
This script will collect the keypoints from your camera. press 'k' to enter the mode to save key points that show 'Record keypoints mode' then press '0-9' as label. the key points will be added to "model/keypoint_classifier/keypoint.csv". 

# Author
Rattasart Sakunrat

# License 
hand-gesture-recognition-using-mediapipe is under [Apache v2 license](LICENSE).
