import os


## 디렉토리 이름 변경 ##
dir_imgs = "img"

for dir_name in os.listdir(dir_imgs):
    #print(file_name[0:5])
    os.rename(dir_imgs+"/"+dir_name, dir_imgs+"/"+dir_name[0:5])


