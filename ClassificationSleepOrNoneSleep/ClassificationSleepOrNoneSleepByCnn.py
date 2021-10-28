import tensorflow as tf
import numpy as np
import cv2, dlib
from imutils import face_utils

import time
import random
import sys
import os

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

motionNum = 2
keyPointNum = 136
frameNum = 25

testX = tf.constant([[[random.randrange(1, 10) for _ in range(keyPointNum)] for _ in range(frameNum)] for _ in range(motionNum)], dtype = tf.float32) # prame per points
testY = tf.constant([ [random.randrange(0, 2) ]for _ in range(2)]) # sleep = 1, didn`t sleep = 0

print(tf.shape(testX))
print(tf.shape(testY))
# Training Model Define ... VGGNet style 14 Layerts network model
model = Sequential([
    Conv2D(input_shape = (28, 28, 1), kernel_size = (3, 3), filters = 32, padding = 'same', activation = 'relu'),
    Conv2D(kernel_size = (3, 3), filters = 64, padding = 'same', activation = 'relu'),
    MaxPool2D(pool_size = (2, 2)),
    Dropout(rate = 0.5),
    Conv2D(kernel_size = (3, 3), filters = 128, padding = 'same', activation = 'relu'),
    Conv2D(kernel_size = (3, 3), filters = 256, padding = 'valid', activation = 'relu'),
    MaxPool2D(pool_size = (2, 2)),
    Dropout(rate = 0.5),
    Flatten(),
    Dense(units = 512, activation = 'relu'),
    Dropout(rate = 0.5),
    Dense(units = 256, activation = 'relu'),
    Dropout(rate = 0.5),
    Dense(units = 10, activation = 'sigmoid')
    
])
# model.add( Dense( units= 1,  activation='sigmoid') )
model.compile( loss='sparse_categorical_crossentropy', optimizer="adam",
              metrics=['acc'] )  # ! gradinet descent 종류 더 알아보기, sparse_categorical_crossentropy 등등 더 있음
## Moddel Training
h = model.fit( testX, testY, epochs = 10)
model.save(savePath)
model.summary()
