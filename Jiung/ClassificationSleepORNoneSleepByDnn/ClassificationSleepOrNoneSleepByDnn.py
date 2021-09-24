import tensorflow as tf
import numpy as np
import cv2, dlib
from imutils import face_utils

import random
import sys
import os

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model



def train(savePath):
    frameNum = 60
    keyPointNum = 68

    motionNum = 100
    PointPerFrame = tf.constant([[[random.randrange(1, 10) for _ in range(keyPointNum)] for _ in range(frameNum)] for _ in range(motionNum)], dtype = tf.float32) # prame per points
    groundTruth = tf.constant([ random.randrange(0, 2) for _ in range(motionNum)]) # sleep = 1, didn`t sleep = 0


    ## Training Model Define

    model = Sequential()

    model.add( Flatten( input_shape= (60,68)))
    model.add( Dense( units= 64,  activation='relu') )
    model.add( Dense( units= 10,  activation='softmax') )
    model.compile( loss='sparse_categorical_crossentropy', optimizer="adam",
                  metrics=['acc'] )  # ! loo종류 더 알아보기, sparse_categorical_crossentropy 등등 더 있음

    ## Moddel Training
    h = model.fit( PointPerFrame, groundTruth, epochs = 100)
    model.save(savePath)
    model.summary()
    

    

def test(model):

    model = load_model(model)
    print("\n")
    model.summary()
    print("\n")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    keyPointModel = ('models/2018_12_17_22_58_35.h5')

    cap = cv2.VideoCapture(0)

    testPointPerFrames = []
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        img = frame.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_img, 1)
        
        pointFrame = []
        for face in faces:
            shapes = predictor(gray_img, face)
            shapes = face_utils.shape_to_np(shapes)
            

            for point in shapes:
                cv2.circle(img, point, 5, (255, 255, 0))

            for keyPoints in shapes:
                for keyPoint in keyPoints:
                    pointFrame.append(keyPoint)

        if len(pointFrame) == 136: # landmark * 2 = 136
            testPointPerFrames.append(pointFrame)
            
        if len(testPointPerFrames) == 10:
            testPointPerFrames = tf.constant(testPointPerFrames, dtype = tf.float32)
            model.predict(testPointPerFrames).argmax(axis = 1)
            testPointPerFrames = []

        


        cv2.imshow('result', img)
        if cv2.waitKey(1) == ord('q'):
            break
    
    

if __name__=="__main__":

    model = ""
    savePath = ""

    if len(sys.argv) == 1:
        print("명령 프롬프트로 실행하세요")
        exit(0)
    
    elif sys.argv[1] == "train": # train save_path
        savePath = sys.argv[2]
        train(savePath)
    elif sys.argv[1] == "test": # test model_path 
        model = f"{sys.argv[2]}"

        test(model)

    else :
        print("잘못 된 명령어 입니다.")
