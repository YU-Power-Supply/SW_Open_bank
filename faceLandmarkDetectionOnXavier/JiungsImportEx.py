import facetracker_custom as fc
import numpy as np
from tensorflow.keras.models import load_model

#yield함수, fps15로 2초마다 68*30프레임 numpy배열 뽑아줌



def classificationSleepOrNonsleep(Y_Data_30_Frames):
    Y_Data_30_Frames = tf.constant(Y_Data_30_Frames, dtype = tf.float32)
    print(Y_Data_30_Frames)



if __name__=="__main__":
    model = "output.h5"
    #model = load_model(model)
    #model.summary() # model Info

    for Y_Data_30_Frames in fc.run(visualize=1, max_threads=4, capture=0):
        classificationSleepOrNonsleep(Y_Data_30_Frames)
    

    
