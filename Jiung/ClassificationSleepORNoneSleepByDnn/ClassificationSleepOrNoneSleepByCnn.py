import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.datasets.fashion_mnist import load_data

# %%
(x_train, y_train), (x_test,y_test) = load_data()

# %%
frameNum = 9
keyPointNum = 9

motionNum = 2

PointPerFrame = tf.constant([[[[2, 4] for _ in range(frameNum)] for _ in range(keyPointNum)] for _ in range(motionNum)], dtype = tf.float32) # 프레임당 좌표
groundTruth = tf.constant([[0], [1]]) # sleep = 1, didn`t sleep = 0


PointPerFrame # 텐서 객체로 잘 변환 되었나 확인
#groundTruth # 결과 값도 잘 변환 되었나 확인

# %%

model = Sequential()

model.add( Flatten( input_shape= (28,28)))
model.add( Dense( units= 64,  activation='relu') )
model.add( Dense( units= 10,  activation='softmax') )
model.compile( loss='sparse_categorical_crossentropy', optimizer="adam",
              metrics=['acc'] )

## loss 종류 알아보기 
## sparse_categorical_crossentropy 등등 더 있음

# %%
h = model.fit( x_train, y_train, epochs = 10)

