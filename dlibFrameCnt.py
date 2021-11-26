import tensorflow as tf
import numpy as np
import cv2, dlib
from imutils import face_utils
import time, random, sys, os, copy, time
from tensorflow.keras.layers import Dense, Dropout , Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    keyPointModel = ('models/2018_12_17_22_58_35.h5')

    cap = cv2.VideoCapture(0)
    startTime = time.time() ### 시간측정
    frameCnt = 0 ### 프레임 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = frame.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_img, 1)
        
        for face in faces:
            shapes = predictor(gray_img, face)
            shapes = face_utils.shape_to_np(shapes)
            
            for point in shapes:
                cv2.circle(img, point, 5, (255, 255, 0))

        presentTime = time.time() - startTime ## 현재시간 측정
        frameCnt += 1 ## 프레임 ++

        if presentTime /1 == 0: ## 1초에 한번
            print(f"FPS : {frameCnt}") ## 프레임 출력 후 초기화
            frameCnt = 0
            
        cv2.imshow('result', img)
        if cv2.waitKey(1) == ord('q'):
            break



if __name__ == '__main__':
    main()
