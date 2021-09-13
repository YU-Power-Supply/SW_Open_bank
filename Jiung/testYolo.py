
import cv2
import numpy as np

def motionAnalysis(x1, y1, x2, y2):
    return 1 if (abs((ord(y2) - ord(y1))) > 1) else 0 

def main(video_path):
    ## version check 
    # print(cv2.__version__)

    VideoSignal = cv2.VideoCapture(video_path) 
    YOLO_net = cv2.dnn.readNet("yolov2-tiny.weights","yolov2-tiny.cfg")   
    classes = [] 
    with open("yolo.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = YOLO_net.getLayerNames() 

    output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]
    

    motionAnalysisByQueue = []
    while True:
        # cam_frame
        ret, frame = VideoSignal.read()
        frame = cv2.resize(frame, (640,480)) # w,h
        h, w, c = frame.shape # h = 720, w = 1280, c = 3

        # yoloNetwork
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0),
        True, crop=False)  
        YOLO_net.setInput(blob) 
        outs = YOLO_net.forward(output_layers) 
        
        class_ids = []
        confidences = []
        boxes = []

        # eliminateLowConfidence
        for out in outs: #outs = [ [...], [...], [...] ]
            for detection in out: # len(outs[0]) = 8112, len(outs[1]) = 2028, len(outs[2]) = 507
                scores = detection[5:] #confidence score 출력 그 앞쪽은 [x][y][w][h][box confidence .. iour 아니면 iou * classconfidence 라고 예상]
                class_id = np.argmax(scores) # take index has highest confidence about each classes
                confidence = scores[class_id] # highest confidence score
                if confidence > 0.5: 
                    # Object detected
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    dw = int(detection[2] * w)
                    dh = int(detection[3] * h)
                    # Rectangle coordinate
                    x = int(center_x - dw / 2)
                    y = int(center_y - dh / 2)
                    boxes.append([x, y, dw, dh]) 
                    confidences.append(float(confidence))
                    class_ids.append(class_id) 

        #print ("boxes : ",boxes, "confidences : ",confidences )
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4) # box indexes
        #print ("indexes : ", indexes) 

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                score = confidences[i]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                cv2.putText(frame, str(label)+"["+str((score*1000)//10)+"%"+"]", (x, y-20), cv2.FONT_ITALIC, 0.5, 
                (255, 255, 255), 1)

                if label == 'person': 
                    motionAnalysisByQueue.append('.'.join([f'{x}',f'{y}',f'{w}',f'{h}']))

                    if (len(motionAnalysisByQueue)==3): # Don`t take more than two frames
                        motionAnalysisByQueue.pop(0)

    
        print("len(motionAnalysisByQueue) : ", len(motionAnalysisByQueue))
        print("motionAnalysisByQueue : ", motionAnalysisByQueue)
        
        if (len(motionAnalysisByQueue)==2) and motionAnalysis(motionAnalysisByQueue[0][0], motionAnalysisByQueue[0][1], motionAnalysisByQueue[1][0], motionAnalysisByQueue[1][1]):
            cv2.putText(frame, "detectedMovindObject!" , (40, 80), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

        cv2.imshow("YOLOv2-tiny", frame)
        #비디오 종료
        if cv2.waitKey(100) == ord("q"):
            break


if __name__ == '__main__':
    video_path = 0
    main(video_path)
