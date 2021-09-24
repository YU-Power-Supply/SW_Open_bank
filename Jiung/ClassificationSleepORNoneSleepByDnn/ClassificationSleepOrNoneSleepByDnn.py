import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
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
    PointPerFrame = tf.constant([[[random.randrange(1, 10) for _ in range(frameNum)] for _ in range(keyPointNum)] for _ in range(motionNum)], dtype = tf.float32) # prame per points
    groundTruth = tf.constant([ random.randrange(0, 2) for _ in range(motionNum)]) # sleep = 1, didn`t sleep = 0


    ## Training Model Define

    model = Sequential()

    model.add( Flatten( input_shape= (68,60)))
    model.add( Dense( units= 64,  activation='relu') )
    model.add( Dense( units= 10,  activation='softmax') )
    model.compile( loss='sparse_categorical_crossentropy', optimizer="adam",
                  metrics=['acc'] )  # ! loo종류 더 알아보기, sparse_categorical_crossentropy 등등 더 있음

    ## Moddel Training
    h = model.fit( PointPerFrame, groundTruth, epochs = 100)
    model.save(savePath)
    model.summary()

def test(model, testPointPerFramePath):
    
    with open(f"{testPointPerFramePath}", "r") as f:
        print(f.readline())
        f.close()
        
    model = load_model(model)
    model.summary()

    # model.predict(testPointPerFrame).argmax(axis = 1)


if __name__=="__main__":

    model = ""
    testPointPerFrame = ""
    savePath = ""

    if len(sys.argv) == 1:
        print("명령 프롬프트로 실행하세요")
        exit(0)
    
    elif sys.argv[1] == "train": # train save_path
        savePath = sys.argv[2]
        train(savePath)
    elif sys.argv[1] == "test": # test model_path testFrame_path
        model = f"{sys.argv[2]}"
        testPointPerFramePath = f"{sys.argv[3]}"

        test(model,testPointPerFramePath)

    else :
        print("잘못 된 명령어 입니다.")
