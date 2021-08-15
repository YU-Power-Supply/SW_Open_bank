import os


## file 이름 변경 ##
file_path = "img"
dir = "img"
for dir_name in os.listdir(dir):
    for file_name in os.listdir(dir +"/"+dir_name):
       os.rename(dir+"/"+dir_name+"/"+file_name, dir+"/"+dir_name+"/"+(file_name[0:6] + file_name[11:14] + file_name[23:])) 

