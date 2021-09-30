import tensorflow as tf
import numpy as np
import cv2, dlib
from imutils import face_utils

import time
import random
import sys
import os

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

keyPointModel = ('models/2018_12_17_22_58_35.h5')

img = cv2.imread('Motions/001_02/Q_001_30_M_02_M0_G0_C0_01.jpg')
faces = detector(img, 1)

for face in faces:
    shapes = predictor(img, face)
    shapes = face_utils.shape_to_np(shapes)
    for point in shapes:
        cv2.circle(img, point, 5, (255, 255, 0))

cv2.imshow('result', img)
cv2.waitKey(0)
