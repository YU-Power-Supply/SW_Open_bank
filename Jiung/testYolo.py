# !/home/jiung/anaconda3/bin/python3 

import cv2
import numpy as np
from collections import deque

def motionAnalysis(x1, y1, x2, y2):
    return 1 if (abs((ord(y2) - ord(y1))) > 1) else 0 

def main(video_path):
    ## version check 
    # print(cv2.__version__)

    VideoSignal = cv2.VideoCapture(video_path) # 웹캠 신호 받기
    YOLO_net = cv2.dnn.readNet("yolov2-tiny.weights","yolov2-tiny.cfg") # YOLO 가중치 파일과 CFG 파일 로드   
    classes = [] # YOLO NETWORK 재구성
    with open("yolo.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = YOLO_net.getLayerNames() # layer의 이름들이 들어가있음 
    layer_names = YOLO_net.getLayerNames() # layer의 이름들이 들어가있음

    output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]
    

    motionAnalysisByQueue = []
    while True:
        # 웹캠 프레임
        ret, frame = VideoSignal.read()
        frame = cv2.resize(frame, (1280,720)) # w,h
        h, w, c = frame.shape # h = 720, w = 1280, c = 3
        
        # image resize
        frame_input = cv2.resize(frame, (720,1280))

        # YOLO 입력
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0),
        True, crop=False) #전처리 # !! 416 x 416 anchor box 사이즈인지 확인 ! 작을수록 빠르고 부정확 
        YOLO_net.setInput(blob) #전처리 후 blob을 네트워크에 전달
        outs = YOLO_net.forward(output_layers) # 정방향 패스가 실행되어 네트워크의 출력으로 예측된 경계 상자 목록을 얻음 이 상자는 신뢰도가 낮은 항목을 필터링 하기 위해 사후 처리 단계를 거침
        

        class_ids = []
        confidences = []
        boxes = []

        #신뢰도가 낮은 바운딩 박스를 제거하기 위한 단계
        for out in outs: #outs = [ [...], [...], [...] ]
            for detection in out: # len(outs[0]) = 8112, len(outs[1]) = 2028, len(outs[2]) = 507
                scores = detection[5:] #confidence score 출력 그 앞쪽은 [x][y][w][h][box confidence .. iou값아니면 iou * classconfidence 라고 예상]
                class_id = np.argmax(scores) # 각 class에 대한 확률중 가장 높은 것의 인덱스 가져옴
                confidence = scores[class_id] # 각 클래스 중에 가장 높은 것의 확률
                if confidence > 0.5: # 컨피던스가 0.5보다 낮은 것 제거
                    # Object detected
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    dw = int(detection[2] * w)
                    dh = int(detection[3] * h)
                    # Rectangle coordinate
                    x = int(center_x - dw / 2)
                    y = int(center_y - dh / 2)
                    boxes.append([x, y, dw, dh]) #cv2로 그림그리기위한 좌표값 반환
                    confidences.append(float(confidence))
                    class_ids.append(class_id) #가장 컨피던스가 높았던 객체의 인덱스 추가


        #print ("boxes : ",boxes, "confidences : ",confidences )
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4) # box indexes
        #print ("indexes : ", indexes) 


        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                score = confidences[i]

                # 경계상자와 클래스 정보 이미지에 입력
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                cv2.putText(frame, str(label)+"["+str((score*1000)//10)+"%"+"]", (x, y-20), cv2.FONT_ITALIC, 0.5, 
                (255, 255, 255), 1)

                if label == 'person': # 사람이라면 바운딩박스 데이터 추가
                    motionAnalysisByQueue.append('.'.join([f'{x}',f'{y}',f'{w}',f'{h}']))

                    if (len(motionAnalysisByQueue)==3): # 2프레임 초과시 과거에것 부터 제거
                        motionAnalysisByQueue.pop(0)

    
        print("len(motionAnalysisByQueue) : ", len(motionAnalysisByQueue))
        print("motionAnalysisByQueue : ", motionAnalysisByQueue)
        
        if (len(motionAnalysisByQueue)==2) and motionAnalysis(motionAnalysisByQueue[0][0], motionAnalysisByQueue[0][1], motionAnalysisByQueue[1][0], motionAnalysisByQueue[1][1]):
            cv2.putText(frame, "detectedMovindObject!" , (40, 80), cv2.FONT_ITALIC, 3, (0, 0, 255), 5)

        cv2.imshow("YOLOv2-tiny", frame)
        #비디오 종료
        if cv2.waitKey(100) > 0:
            break


if __name__ == '__main__':
    video_path = 0
    main(video_path)
