import tensorflow as tf
import numpy as np
import cv2, dlib
from imutils import face_utils

import time, random, sys, os, copy

import tensorflow
import imutils
from tensorflow.keras.layers import Dense, Dropout , Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model




def train(trainDataPath, saveModelPath):
    pointPerFramePerMotion = []
    frameNum = 30
    keyPointNum = 68


    
    # read trainData
    groundTruth = []
    for motion in os.listdir(trainDataPath):
        pointPerFrame = []
        if motion[-4:]== ".txt":
            file = open(f"{trainDataPath}/{motion}", "r", encoding = 'UTF8')
            groundTruth.append(int(file.readline()))

            for frames in file:
                pointX = frames.split()
                if len(pointX) != 68: print(f"{file}") # CHECK 68 POINT
                pointX = list(map(int, pointX)) # convert str to integer
                pointPerFrame.append(pointX)
            pointPerFramePerMotion.append(pointPerFrame)


    pointPerFramePerMotion = tf.constant(pointPerFramePerMotion, dtype = tf.float32) # prame per points
    print("pointPerFrameMotion`s dims : ", tf.shape(pointPerFramePerMotion))

    groundTruth = tf.constant(groundTruth, dtype = tf.float32) # sleep = 1, didn`t sleep = 0
    print("groundTruth`s dims : ", tf.shape(groundTruth))

    # '''
    # testX = tf.constant([[[random.randrange(1, 10) for _ in range(keyPointNum)] for _ in range(frameNum)] for _ in range(2)], dtype = tf.float32) # prame per points
    # testY = tf.constant([ [random.randrange(0, 2) ]for _ in range(2)]) # sleep = 1, didn`t sleep = 0
    # 
    # print(tf.shape(testX))
    # print(tf.shape(testY))
    # '''


    # Training Model Define
    model = Sequential([
        Flatten(input_shape= (frameNum,keyPointNum)), 
        Dense(units= 64,  activation='relu'),
        Dropout(0.2),
        Dense(units= 32,  activation='relu'),
        Dropout(0.2),
        Dense(units = 2, activation = 'softmax')
    ])

    # model.add( Dense( units= 1,  activation='sigmoid') )
    model.compile( loss='sparse_categorical_crossentropy', optimizer="adam",
                  metrics=['acc'] )  # ! gradinet descent 종류 더 알아보기, sparse_categorical_crossentropy 등등 더 있음

    ## Moddel Training
    h = model.fit( pointPerFramePerMotion, groundTruth, epochs = 1000)
    model.save(saveModelPath)
    model.summary()


def mkdirs(dirPath, fromNum, toNum):
    for i in range(int(fromNum), int(toNum) +1):
        try:
            os.makedirs(f"{dirPath}/{i}")
        except OSError:
            print ('Error: Creating directory.')


def dataPreprocessing(imgPath, fromMotion, toMotion):
    
    # face detect && find keypoints on landmark 
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    keyPointModel = ('models/2018_12_17_22_58_35.h5')

    
    groundTruth = "1"
    motionCnt = 0
 
    for motion in range(int(fromMotion),int(toMotion)+1 ):
        pointPerFrame = []
        motionCnt += 1
        frameCnt = 0
        
        with open(f"{imgPath}/{motion}.txt", "w", encoding = 'UTF8') as f:
            f.write(groundTruth)

        frameList = []
        for frame in os.listdir(f"{imgPath}/{motion}"):
            frameList.append(frame)
        frameList.sort()

        for frame in frameList: 
            img = cv2.imread(f"{imgPath}/{motion}/{frame}")
            faces = detector(img, 1)
            
            if(len(faces) == 0) : 
                print(f"\n img : {frame}, couldn`t detect this face")
            else :
                frameCnt += 1
                shapes = predictor(img, faces[0])
                shapes = face_utils.shape_to_np(shapes)


                with open(f"{imgPath}/{motion}.txt", "a", encoding = 'UTF8') as f:
                    f.write("\n")
                    for pointY in shapes[:,1]:
                        f.write(str(pointY) + " ")
                
        for _ in range(30-frameCnt):
            with open(f"{imgPath}/{motion}.txt", "a", encoding = 'UTF8') as f:
                    f.write("\n")
                    for pointY in shapes[:,1]:
                        f.write(str(pointY) + " ")
        print(f"motion[{motion}] : detected total {frameCnt} motions \ntotal {motionCnt} motions completed ")
        
                        
if __name__=="__main__":
    if len(sys.argv) == 1:
        print("명령 프롬프트로 실행하세요")
        exit(0)
    elif sys.argv[1] == "mkdirs": # dirPath fromNum toNum 
        mkdirs(sys.argv[2], sys.argv[3], sys.argv[4])

    elif sys.argv[1] == "train": # [train] [trainDataPath] [saveModelPath]
        train(sys.argv[2], sys.argv[3])

    elif sys.argv[1] == 'data': # [data] [imgsPath] [fromMotion] [toMotion]
        dataPreprocessing(sys.argv[2], sys.argv[3], sys.argv[4])

    else :
        print("잘못 된 명령어 입니다.")
