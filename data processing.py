import os
import cv2
import json

path = './json/R_202_40_M_01_M0_G0_C0_01.json'
imgpath = './img/R_202_40_M_01_M0_G0_C0_01.jpg'
path2 = './'

imgx = 720
imgy = 1280

img = cv2.imread(imgpath)
img2 = cv2.imread(imgpath)
img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, (1280, 720), interpolation=cv2.INTER_AREA)


marking = [(0.473828, 0.604861, 0.471094, 0.473611),
        (0.355859, 0.590972, 0.085156, 0.048611),
        (0.528516, 0.588194, 0.100781, 0.048611),
        (0.432031, 0.718750, 0.156250, 0.062500)]

data = ''
with open(path) as file:
    json_data = json.load(file)
    bounding_data = json_data['ObjectInfo']['BoundingBox']
    for idx, obj in enumerate(bounding_data.values()):
        if obj['isVisible']:
            ltx, lty, rbx, rby = map(float, obj['Position'])
            ltx = ltx/imgx*imgy
            rbx = lty/imgx*imgy
            lty = lty/imgy*imgx
            rby = rby/imgy*imgx

            img = cv2.rectangle(img, (int(ltx), int(lty)), (int(rbx), int(rby)), (0, 0, 255), thickness= 2)
            
            #midx = ((rbx+ltx)/2)/imgx
            #midy = ((lty + rby)/2)/imgy
            #width = (rbx - ltx)/imgx
            #hight = (rby - lty)/imgy
            
            print("json : ", ltx, lty, rbx, rby)
            
            
for mark in marking:
        midx, midy, width, length = map(lambda x : round(x, 2), mark)
        ltx = (midx - width/2)*imgy
        rbx = (midx + width/2)*imgy
        lty = (midy - length/2)*imgx
        rby = (midy + length/2)*imgx
        print("yolo : ", ltx, lty, rbx, rby)
        img2 = cv2.rectangle(img2, (int(ltx), int(lty)), (int(rbx), int(rby)), (0, 0, 255), thickness= 2)


        #ltx = (midx - width/2)*imgx
        #rbx = (midx + width/2)*imgx
        #lty = (midy - length/2)*imgy
        #rby = (midy + length/2)*imgy
            #conversion x, y, w, h
            #transpos = '{} {} {} {}'.format(((ltx + rbx)/2)/720, ((lty + rby)/2)/1280, (rbx - ltx)/720, (rby - lty)/1280)
            #data += f"{idx} {transpos}\n"

cv2.imshow('img', img)
cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
                

#face, leye, reye, mouth, ciger, phone => class = 6
#img 720 X 1280 => %
