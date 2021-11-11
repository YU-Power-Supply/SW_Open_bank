import facetracker_custom as fc
import numpy as np


#yeild함수, fps15로 2초마다 68*30프레임 numpy배열 뽑아줌

for f in fc.run(visualize=1, max_threads=4, capture="video.mp4"):
    
    print(f)
