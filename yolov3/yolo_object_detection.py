import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
INPUT_SIZE = (720, 1280)

# 눈 깜빡임 대신 임시로 사용
def cellphone_detect(cellphone_con, cellphone_score):
    print('continuation : ', cellphone_con)
    if cellphone_con == 0:
        cellphone_score = 1
        cellphone_con = 1
    else:
        cellphone_score += 1

    if cellphone_score > 10: #초당 프레임 * 내가 설정한 시간(초) 여기서는 2프레임 * 5초
        print('danger! do not use cellphone!!')
    print('score : ', cellphone_score)

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    frame = 0
    cellphone_con = 0
    cellphone_score = 0
    start_time = time.time()

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(image, INPUT_SIZE)
        height, width, channels = image.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False) # 전처리

        net.setInput(blob) # 네트워크에 입력
        outs = net.forward(output_layers) # 결과 output

        # Showing informations on the screen
        class_ids = []
        confidences = []
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

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # 노이즈 제거(겹쳐져 있는 박스들 중 확률이 가장 높은 박스만 남김)
        font = cv2.FONT_HERSHEY_PLAIN

        labels = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                labels.append(label)
                color = colors[class_ids[i]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label, (x, y + 30), font, 3, color, 3)

        if 'cell phone' in labels:
            if cellphone_con == 0:
                cellphone_score = 1
                cellphone_con = 1
            else:
                cellphone_score += 1

            if cellphone_score > 10:  # 초당 프레임 * 내가 설정한 시간(초) 여기서는 2프레임 * 5초
                print('danger! do not use cellphone!!')
            print('score : ', cellphone_score)
        else:
            cellphone_con = 0

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Image', image)
        frame += 1
        if cv2.waitKey(1) == ord('q'):
            end_time = time.time()
            print('running time : ', end_time - start_time)
            print('Frame : ', frame)
            break

if __name__ == '__main__':
    video_path = 0
    main(video_path)
