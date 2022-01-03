#--V 1.0--
import os
import sys
import cv2
import json
import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree, dump

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def face_detection(img_path, net, output_layers):

    img = cv2.imread(img_path)
    height = 1280
    width = 720
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 정보를 화면에 표시
    class_ids = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                class_ids.append(class_id)
    
    return boxes[class_ids.index([0])] # face 의 index가 0번이므로 0번 리스트만 반환


def json_to_pts(json_path, save_path):
    file_list = os.listdir(json_path)
    
    for i in file_list:
        data = 'version: 1\nn_points:  70\n{\n'
        with open(json_path+i, encoding='UTF8') as file:
            json_data = json.load(file)
            landmark_data = json_data['ObjectInfo']['KeyPoints']['Points']
            for n in range(70):
                data += ' '.join([landmark_data[2*n], landmark_data[2*n+1]]) + '\n'
            data += '}'
        
        #데이터 저장부
        savepath = save_path + i[:-4] + "pts" # json -> txt
        with open(savepath, "w", encoding='UTF8') as wfile:
            wfile.write(data)


def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def make_xml(json_path, img_path, out_path = None, parts = None):
    file_list = os.listdir(json_path)

    #---xml setting---
    for i in file_list:
        imgpath = img_path + i[:-4] + "jpg"
        left, top, width, height = face_detection(imgpath, net, output_layers)
        
        node1 = Element("image", file = imgpath, width = '720', height = '1280')
        node2 = Element("box", top=str(top), left=str(left), width=str(width), height=str(height)) # yolo detecting box 넣기
        root.append(node1)
        node1.append(node2)

        with open(json_path + i, encoding='UTF8') as file:
            json_data = json.load(file)
            landmark_data = json_data['ObjectInfo']['KeyPoints']['Points']
            for n in range(70):
                node3 = Element("part", name=str(n).zfill(2), x=str(round(float(landmark_data[2*n]))-1), y=str(round(float(landmark_data[2*n+1]))-1))
                node2.append(node3)


    
    
def remove_pts(ptspath):  # when you want to remove txt files
    for t in os.listdir(ptspath):
        if t[-3:] == "pts":
            os.remove(ptspath+t)


if __name__ == '__main__':
    
    img_dir = 'Validation/Landmarkimg/' #route of img_dir
    json_dir = 'Validation/Landmark/'    #route of json_dir

    img_dir_list = list(map(lambda x : img_dir + x + "/", os.listdir(img_dir)))
    json_dir_list = list(map(lambda x : json_dir + x + "/", os.listdir(json_dir)))

    img_dir_list.sort()
    json_dir_list.sort()
    
    
    #---yolo setting---
    net = cv2.dnn.readNet("yolo-obj_final.weights", "yolo-obj.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    root = Element("images")
    
    if len(sys.argv) == 1:
        print("명령 프롬프트로 실행하세요")
        exit(0)

    elif sys.argv[1] == "jtp": # json to text
        for i in range(207, len(json_dir_list)):
            print(i)
            json_to_pts(json_dir_list[i], img_dir_list[i])

    elif sys.argv[1] == "xml": # json to text
        for i in range(100):
            print(i)
            make_xml(json_dir_list[i],img_dir_list[i], net, output_layers)
            
        indent(root)
        ElementTree(root).write("valid.xml", encoding='ISO-8859-1', xml_declaration=True)

    elif sys.argv[1] == "rmv": #remove
        for i in img_dir_list:
            print(i)
            remove_pts(i)

    print("완료")