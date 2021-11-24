import facetracker_custom as fc
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


#yield함수, fps15로 2초마다 68*30프레임 numpy배열 q뽑아줌



def classificationSleepOrNonsleep(Y_Data_30_Frames):
    #print(Y_Data_30_Frames)
    ## tensor generate
    Y_Data_30_Frames = tf.expand_dims(Y_Data_30_Frames, axis = 0)
    Y_Data_30_Frames = tf.expand_dims(Y_Data_30_Frames, axis = 3)

    h = model.predict(Y_Data_30_Frames)
   

    print(f"Sleep [{(h[0][1]*100)//1}%]") if h.argmax() == 1 else print(f"NO Sleep [{(h[0][0])*100//1}%]")
    

if __name__=="__main__":
    model = "output.h5"
    model = load_model(model)
    model.summary() # model Info

    for Y_Data_30_Frames in fc.run(visualize=1, max_threads=4, capture=0):
        classificationSleepOrNonsleep(Y_Data_30_Frames)
    

    
