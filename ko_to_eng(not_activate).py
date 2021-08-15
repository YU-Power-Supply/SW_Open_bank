# import os
# import glob
# 
# path = "./*"
# #img_list = os.listdir(path)
# img_list = glob.glob(path)
# img_list_py = [file for file in img_list if file.endswith(".py")]
# 
# print("img_list : {}".format(img_list_py))

import os
import fnmatch

i = "지웅"

path = os.getcwd()
for file_name in os.listdir(path):
	if fnmatch.fnmatch(file_name, "*지웅*"): print("okay") if (i==file_name) else print("no")
