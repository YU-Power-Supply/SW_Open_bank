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
    
    pointPerFramePerMotion = []
    frameNum = 25
    keyPointNum = 136

    ## read data
    # read sleep data
    for file in os.listdir(sleepPath):
        pointPerFrame = []
        if file[-10:]== "_train.txt":
            file = open(f"{sleepPath}/{file}", "r", encoding = 'UTF8')
            for frames in file:
                keyPoints = frames.split()
                keyPoints = list(map(int, keyPoints)) # convert str to integer
                pointPerFrame.append(keyPoints)
            pointPerFramePerMotion.append(pointPerFrame)
            

    # read nonsleep data
    for file in os.listdir(nonSleepPath):
        pointPerFrame = []
        if file[-10:]== "_train.txt":
            file = open(f"{nonSleepPath}/{file}", "r", encoding = 'UTF8')
            for frames in file:
                keyPoints = frames.split()
                keyPoints = list(map(int, keyPoints)) # convert str to integer
                pointPerFrame.append(keyPoints)
    
            pointPerFramePerMotion.append(pointPerFrame)

    groundTruth = []
    # read groundTruth sleep data
    for file in os.listdir(sleepPath):
        if file[-10:]== "ground.txt":
            with open(f"{sleepPath}/{file}", "r", encoding = 'UTF8') as f:
                groundTruth.append(int(f.read()))
                

    # read groundTruth nonsleep data
    for file in os.listdir(nonSleepPath):
        if file[-10:]== "ground.txt":
            with open(f"{sleepPath}/{file}", "r", encoding = 'UTF8') as f:
                groundTruth.append(int(f.read()))
                


    pointPerFramePerMotion = tf.constant(pointPerFramePerMotion, dtype = tf.float32) # prame per points
    # print("pointPerFrameMotion`s dims : ", tf.shape(pointPerFramePerMotion))

    groundTruth = tf.constant(groundTruth, dtype = tf.float32) # sleep = 1, didn`t sleep = 0
    print("groundTruth`s dims : ", tf.shape(groundTruth))
# 
    # '''
    # testX = tf.constant([[[random.randrange(1, 10) for _ in range(keyPointNum)] for _ in range(frameNum)] for _ in range(2)], dtype = tf.float32) # prame per points
    # testY = tf.constant([ [random.randrange(0, 2) ]for _ in range(2)]) # sleep = 1, didn`t sleep = 0
    # 
    # print(tf.shape(testX))
    # print(tf.shape(testY))
    # '''
# 
# 
   # 
# 
# 
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


def dataPreprocessingMakeDir(sleepPath, nonSleepPath, dirPath, dir):
    try:
        if not os.path.exists(f"{sleepPath}"): # if you don't have same dir, create
            os.makedirs(f"{sleepPath}")
    except OSError:
        print ('Error: Creating directory. ' +  f"{sleepPath}")

    try:
        if not os.path.exists(f"{nonSleepPath}"): # if you don't have same dir, create
            os.makedirs(f"{nonSleepPath}")
    except OSError:
        print ('Error: Creating directory. ' +  f"{nonSleepPath}")
        
    sleepScenarios = ["02", "03", "09", "10", "16", "17", "21", "22", "28", "29"]
    
    for dir in os.listdir(dirPath):
        if (dir[11:13]) in sleepScenarios:
            os.rename(f"{dirPath}/{dir}", f"{sleepPath}/{dir}")
        else:
            os.rename(f"{dirPath}/{dir}", f"{nonSleepPath}/{dir}")


def dataPreprocessing(sleepPath, nonSleepPath, dirPath):
    

    dirChecker = []
    for dir in os.listdir(os.getcwd()):
        dirChecker.append(dir)

    # Labeling data by directory 
    # if already there are sleep && nonsleep dir this method don`t operate
    if (sleepPath not in dirChecker) or (nonSleepPath not in dirChecker):
        dataPreprocessingMakeDir(sleepPath, nonSleepPath, dirPath, dir)

    

    # face detect && find keypoints on landmark 
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    keyPointModel = ('models/2018_12_17_22_58_35.h5')

    
    imgCnt = 0
    motionCnt = 0
    frameNum = 25
    keyPointNum = 136

    for dir in os.listdir(sleepPath): # sleep images
        groundTruth = []

        if not (dir[-4:] == ".txt"):
            pointPerFrame = []
            motionCnt += 1

            groundTruth.append("1") # sleep : True
            for file in os.listdir(f"{sleepPath}/{dir}"):
                img = cv2.imread(f'{sleepPath}/{dir}/{file}')
                faces = detector(img, 1)
                points = []

                #얼굴이 detect되지 않았을 때
                if (len(faces) == 0) : print(f"\nimg : {file},  couldn`t detect present face ")
                else: 
                    for face in faces:
                        shapes = predictor(img, face)
                        shapes = face_utils.shape_to_np(shapes)
                        for keyPointXY in shapes:
                            for keyPoint in keyPointXY:
                                points.append(keyPoint)

                    imgCnt += 1
                    print(f" img : {file} , The number of completed [sleep image] : {imgCnt} / 41053") # image checker
                    pointPerFrame.append(points)
                              
                            
            for _ in range(frameNum - len(pointPerFrame)): # detect 되지 않은것이 있을 때
                pointPerFrame.append(pointPerFrame[-1])
                imgCnt += 1
            print(f"The number of completed [sleep] motion : {motionCnt} / 1658") # motion checker

            # convert integer to str 
            for i in range(len(pointPerFrame)):
                pointPerFrame[i] = list(map(str, pointPerFrame[i]))

            with open(f"{sleepPath}/{dir}_train.txt", "w", encoding = 'UTF8') as f:
                for frames in pointPerFrame:
                    for points in frames:
                        f.write(points+" ")
                    f.write("\n")

            with open(f"{sleepPath}/{dir}_ground.txt", "w", encoding = 'UTF8') as f:
                for grounds in groundTruth:
                    f.write(grounds+ " ")  



    for dir in os.listdir(nonSleepPath): # nonsleep images
        groundTruth = []

        if not (dir[-4:] == ".txt"):
            pointPerFrame = []
            motionCnt += 1

            groundTruth.append("0") # nonsleep : True
            for file in os.listdir(f"{nonSleepPath}/{dir}"):
                img = cv2.imread(f'{nonSleepPath}/{dir}/{file}')
                faces = detector(img, 1)
                points = []
                

                if (len(faces) == 0) : print(f"\nimg : {file},  couldn`t detect present face ")
                else :
                    for face in faces:
                        shapes = predictor(img, face)
                        shapes = face_utils.shape_to_np(shapes)
                        for keyPointXY in shapes:
                            for keyPoint in keyPointXY:
                                points.append(keyPoint)

                    imgCnt += 1
                    print(f" img : {file} , The number of completed [sleep image] : {imgCnt} / 41053") # image checker
                    pointPerFrame.append(points)

            for _ in range(25 - len(pointPerFrame)): # detect 되지 않은것이 있을 때
                pointPerFrame.append(pointPerFrame[-1])
                imgCnt += 1
            print(f"The number of completed [nonsleep] motion : {motionCnt} / 1658") # motion checker

            for i in range(len(pointPerFrame)):
                pointPerFrame[i] = list(map(str, pointPerFrame[i]))

            with open(f"{nonSleepPath}/{dir}_train.txt", "w", encoding = 'UTF8') as f:
                for frames in pointPerFrame:
                    for points in frames:
                        f.write(points+" ")
                    f.write("\n")

            with open(f"{nonSleepPath}/{dir}_ground.txt", "w", encoding = 'UTF8') as f:
                f.write(groundTruth)

       

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
    elif sys.argv[1] == 'data': # [data] [saveSleepPath] [saveNonSleepPath] [originImgPath]
        dataPreprocessing(sys.argv[2], sys.argv[3], sys.argv[4])

    else :
        print("잘못 된 명령어 입니다.")
