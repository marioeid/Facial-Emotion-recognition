import csv
import copy
import itertools
#Import necessary libraries
from flask import Flask, render_template, Response
from flask_restful import Resource, Api
from flask_cors import CORS

#Initialize the Flask app
app = Flask(__name__)

CORS(app)

api=Api(app)

import cv2 as cv
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_info_text(image, brect, facial_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if facial_text != "":
        info_text = 'Emotion :' + facial_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image



# Model load
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 

keypoint_classifier = KeyPointClassifier()


# Read labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]
import cv2

'''mode = 0
video_cap = cv2.VideoCapture(0)

frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_cap.get(cv2.CAP_PROP_FPS))
frameTime = 10 # time of each frame in ms, you can add logic to change this value.
fps=fps
# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('with_report.mp4', fourcc, fps, (frame_width, frame_height))
cnt=0
use_brect = True

while True:

    # Process Key (ESC: end)
    key = cv.waitKey(10)
    if key == 27:  # ESC
        break

    # Camera capture
    ret, image = video_cap.read()
    if not ret:
        break
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, face_landmarks)

            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, face_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)

            #emotion classification
            facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
            # Drawing part
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            debug_image = draw_info_text(
                    debug_image,
                    brect,
                    keypoint_classifier_labels[facial_emotion_id])
    
    output.write(debug_image)
    # Screen reflection
    cv.imshow('Facial Emotion Recognition', debug_image)

video_cap.release()
cv.destroyAllWindows()
'''

from deepface import DeepFace
from deepface.detectors import FaceDetector
import pandas as pd


import cv2
import copy
video_cap = cv2.VideoCapture(0)

def gen_frames():

    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    frameTime = 10 # time of each frame in ms, you can add logic to change this value.
    fps=fps
    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('with_report.mp4', fourcc, fps, (frame_width, frame_height))
    cnt=0
    while True:

        key = cv.waitKey(10)
        if key == 27 :  # ESC
            break

        ret, image = video_cap.read()
        if not ret:
            break

        image.flags.writeable = False

        image = cv.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        debug_image=copy.deepcopy(image)
        # Detection implementation
        detector_name="opencv"
        detector = FaceDetector.build_model(detector_name) #set opencv, ssd, dlib, mtcnn or retinaface
        obj2 = FaceDetector.detect_faces(detector, detector_name, image)
        
        if (len(obj2)>=1):
            for i in range(len(obj2)):
                cur=obj2[i][1]
                x=cur[0]
                y=cur[1]
                w=cur[2]
                h=cur[3]
                detected_face=image[y:y+h,x:x+w]
                
                
                image_height, image_width, c = detected_face.shape
                black_image=np.zeros((image_height,image_width, 1), dtype = "uint8")
                results = face_mesh.process(detected_face)
                if results.multi_face_landmarks is not None:
                    for face_landmarks in results.multi_face_landmarks:
                        landmark_list = calc_landmark_list(detected_face, face_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(
                            landmark_list)
                        facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
                        predictions= keypoint_classifier_labels[facial_emotion_id]
                        dict = {'frame_num':cnt,'num_of_faces': len(obj2),'face_num':i,'Emotion':predictions}
                        (wt, ht), _ = cv.getTextSize(predictions, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        cv.rectangle(debug_image,(x,y-40),(x+wt,y),(0,0,0),-1)
                        cv.putText(debug_image, predictions, (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv.rectangle(debug_image,(x,y),(x+w,y+h),(0,0,0),1)
                        for i in range(0,478):
                            pt1=face_landmarks.landmark[i]
                            locx=int(pt1.x*image_height)
                            locy=int(pt1.y*image_width)
                            cv2.circle(detected_face,(locx,locy),2,(100,100,0),-1)
                            cv2.circle(black_image,(locx,locy),2,(255,255,255),-1)
                # Screen reflection
        
        
            #cv.imshow('Facial Emotion Recognition2', detected_face)
            #cv.imshow('Facial Emotion Recognition3', black_image)
            #cv.imshow('Facial Emotion Recognition', debug_image)
            ret, buffer = cv2.imencode('.jpg', detected_face)
            detected_face=buffer.tobytes()
            
            ret, buffer = cv2.imencode('.jpg', black_image)
            detected_face=buffer.tobytes()

            ret, buffer = cv2.imencode('.jpg', debug_image)
            debug_image=buffer.tobytes()

            #yield (b'--frame\r\n'
                   #b'Content-Type: image/jpeg\r\n\r\n' + detected_face + b'\r\n')  # concat frame one by one and show result
            #yield (b'--frame\r\n'
                    #b'Content-Type: image/jpeg\r\n\r\n' + black_image + b'\r\n')  # concat frame one by one and show result
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + debug_image + b'\r\n')  # concat frame one by one and show result
        
         
            cnt+=1
            #if cnt==300:break
            print(cnt)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
    


