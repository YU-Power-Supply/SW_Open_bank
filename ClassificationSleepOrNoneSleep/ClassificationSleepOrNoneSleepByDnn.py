import tensorflow as tf
import numpy as np
import cv2, dlib
from imutils import face_utils

import time, random, sys, os, copy

from tensorflow.keras.layers import Dense, Dropout , Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


# print("dlib", dlib.__version__, "numpy", np.__version__)


def train(sleepPath, nonSleepPath, savePath):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    keyPointModel = ('models/2018_12_17_22_58_35.h5')

    pointPerFramePerMotion = []
    groundTruth = []
    
    imgCnt = 0
    motionCnt = 0

    frameNum = 25
    keyPointNum = 136

    for dir in os.listdir(sleepPath): # sleep images
        pointPerFrame = []
        cnt = 0 # 얼마나 얼굴이 detect되지 않나 카운트
        motionCnt += 1

        groundTruth.append(1)
        for file in os.listdir(f"{sleepPath}/{dir}"):
            img = cv2.imread(f'{sleepPath}/{dir}/{file}')
            faces = detector(img, 1)
            points = []

            if (len(faces) == 0) : #얼굴이 detect되지 않았을 때
                cnt += 1
                print(f"img : {file}, ** couldn`t detect present face **")

            for face in faces:
                shapes = predictor(img, face)
                shapes = face_utils.shape_to_np(shapes)
                for keyPointXY in shapes:
                    for keyPoint in keyPointXY:
                        points.append(keyPoint)
                        if(len(points) == keyPointNum):
                            imgCnt += 1
                            print(f" img : {file} , The number of completed [sleep image] : {imgCnt} / 41053") # image checker
                            pointPerFrame.append(points)
        
                                            
        
        for _ in range(frameNum - len(pointPerFrame)): # detect 되지 않은것이 있을 때
            pointPerFrame.append(pointPerFrame[-1])
            imgCnt += 1
        pointPerFramePerMotion.append(pointPerFrame)
        print("pointPerFrame : ", len(pointPerFrame))
        print(f"The number of completed [sleep] motion : {motionCnt} / 1658") # motion checker
    


    for dir in os.listdir(nonSleepPath): # nonsleep images
        pointPerFrame = []
        cnt = 0 # 얼마나 얼굴이 detect되지 않나 카운트
        motionCnt += 1

        groundTruth.append(0)
        for file in os.listdir(f"{nonSleepPath}/{dir}"):
            img = cv2.imread(f'{nonSleepPath}/{dir}/{file}')
            faces = detector(img, 1)
            points = []

            if (len(faces) == 0) : 
                cnt += 1
                print(f"img : {file}, ** couldn`t detect present face **")

            for face in faces:
                shapes = predictor(img, face)
                shapes = face_utils.shape_to_np(shapes)
                for keyPointXY in shapes:
                    for keyPoint in keyPointXY:
                        points.append(keyPoint)
                        if(len(points) == 136):
                            imgCnt += 1
                            print(f" img : {file} , The number of completed [non sleep] image : {imgCnt} / 41053") # image checker
                            pointPerFrame.append(points)
            

        for _ in range(25 - len(pointPerFrame)): # detect 되지 않은것이 있을 때
            pointPerFrame.append(pointPerFrame[-1])
            imgCnt += 1
        pointPerFramePerMotion.append(pointPerFrame)
        print("pointPerFrame : ", len(pointPerFrame))
        print(f"The number of completed [nonsleep] motion : {motionCnt} / 1658") # motion checker
    
    
                   

    pointPerFramePerMotion = tf.constant(pointPerFramePerMotion, dtype = tf.float32) # prame per points
    print(f"complete convert to tensor , total {cnt} images aren`t detected ")
    print("pointPerFrameMotion`s dims : ", tf.shape(pointPerFramePerMotion))

    groundTruth = tf.constant(groundTruth, dtype = tf.float32) # sleep = 1, didn`t sleep = 0
    print("groundTruth`s dims : ", tf.shape(groundTruth))

    '''
    testX = tf.constant([[[random.randrange(1, 10) for _ in range(keyPointNum)] for _ in range(frameNum)] for _ in range(2)], dtype = tf.float32) # prame per points
    testY = tf.constant([ [random.randrange(0, 2) ]for _ in range(2)]) # sleep = 1, didn`t sleep = 0
    
    print(tf.shape(testX))
    print(tf.shape(testY))
    '''


   


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
    model.save(savePath)
    model.summary()



def test(model):

    model = load_model(model)
    print("")
    model.summary()
    print("")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    keyPointModel = ('models/2018_12_17_22_58_35.h5')

    cap = cv2.VideoCapture(0)

    testPointPerFrames = []
    startTime = time.time()
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
            print("Counted Frame Number", len(testPointPerFrames))
        
        if len(testPointPerFrames) == 25:
            # print(testPointPerFrames)
            testPointPerFrames = tf.constant(testPointPerFrames, dtype = tf.float32)
            
            # print(testPointPerFrames.shape) # Dims
            testPointPerFrames = tf.expand_dims(testPointPerFrames, axis=0)
            
            # print(testPointPerFrames.shape) # reshaped Dims

            print(model.predict(testPointPerFrames).argmax(axis =1))
            endTime = time.time()
            print("Time: " ,endTime - startTime)
            
            testPointPerFrames = []

        


        cv2.imshow('result', img)
        if cv2.waitKey(1) == ord('q'):
            break
                
                
def dataPreprocessing(dirPath, sleepPath, nonSleepPath):
    sleepScenarios = ["02", "03", "09", "10", "16", "17", "21", "22", "28", "29"]

    for dir in os.listdir(dirPath):
        if (dir[11:13]) in sleepScenarios:
            os.rename(f"{dirPath}/{dir}", f"{sleepPath}/{dir}")
        else:
            os.rename(f"{dirPath}/{dir}", f"{nonSleepPath}/{dir}")

    



if __name__=="__main__":

    model = ""
    savePath = ""

    if len(sys.argv) == 1:
        print("명령 프롬프트로 실행하세요")
        exit(0)
    
    elif sys.argv[1] == "train": # [train] [sleepPath] [nonSleepPath] [save_path]
        train(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "test": # [test] [model_path] 
        model = f"{sys.argv[2]}"
        test(model)
    elif sys.argv[1] == 'data': # [data] [dirPath] [saveSleepPath] [saveNonSleepPath]
        dataPreprocessing(sys.argv[2], sys.argv[3], sys.argv[4])

    else :
        print("잘못 된 명령어 입니다.")
