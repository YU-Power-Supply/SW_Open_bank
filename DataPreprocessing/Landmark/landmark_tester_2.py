import cv2 
import dlib
import numpy as np
import time
from math import hypot 

# create default face detector 
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("./ALL_LANDMARKS_68-2.dat") 
mouth_points = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67] 
r_eye_points = [42, 43, 44, 45, 46, 47] 
l_eye_poits = [36, 37, 38, 39, 40, 41] 

def midpoint(p1, p2): 
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2) 

def get_mouth_pen_ratio(mouth_points, facial_landmarks): 
    left_point = (facial_landmarks.part( mouth_points[12]).x, facial_landmarks.part(mouth_points[12]).y) 
    right_point = (facial_landmarks.part( mouth_points[16]).x, facial_landmarks.part(mouth_points[16]).y) 
    center_top = midpoint(facial_landmarks.part( mouth_points[13]), facial_landmarks.part(mouth_points[14])) 
    center_bottom = midpoint(facial_landmarks.part( mouth_points[19]), facial_landmarks.part(mouth_points[18])) 
    hor_line = cv2.line(image, left_point, right_point, (0, 255, 0), 2) 
    ver_line = cv2.line(image, center_top, center_bottom, (0, 255, 0), 2) 
    hor_line_lenght = hypot( (left_point[0] - right_point[0]), (left_point[1] - right_point[1])) 
    ver_line_lenght = hypot( (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1])) 
    if ver_line_lenght != 0: 
        ratio = hor_line_lenght / ver_line_lenght 
    else: 
        ratio = 60 
    return ratio 

def get_blinking_ratio(eye_points, facial_landmarks): 
    left_point = (facial_landmarks.part( eye_points[0]).x, facial_landmarks.part(eye_points[0]).y) 
    right_point = (facial_landmarks.part( eye_points[3]).x, facial_landmarks.part(eye_points[3]).y) 
    center_top = midpoint(facial_landmarks.part( eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part( eye_points[5]), facial_landmarks.part(eye_points[4])) 
    hor_line = cv2.line(image, left_point, right_point, (0, 255, 0), 2) 
    ver_line = cv2.line(image, center_top, center_bottom, (0, 255, 0), 2) 
    hor_line_lenght = hypot( (left_point[0] - right_point[0]), (left_point[1] - right_point[1])) 
    ver_line_lenght = hypot( (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1])) 
    ratio = hor_line_lenght / ver_line_lenght 
    return ratio


if __name__ == "__main__":
    # convert frame to image 

    #save_point = "./1113.jpg" # put your image
    #image = cv2.imread(save_point)
    cap = cv2.VideoCapture('./video.mp4')
    cap.set(3,640)
    cap.set(4,480)
    start = time.time()
    count = 0
    try:
        while True:
            success, img = cap.read()
            image = cv2.flip(img,1)#좌우반전, 0을 넣는다면 상하반전
            if (success):
                count += 1
                faces = detector(image)
                image = np.array(image)

                for face in faces: 
                    
                    #cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0,0,255), 2)
                    land = predictor(image, face)
                    land_list = []
                    for l in land.parts():
                        land_list.append([l.x, l.y])
                        cv2.circle(image, (l.x, l.y), 3, (255,0,0), -1)
                    
                    print(face)
                    
                    landmarks = predictor(image, face)
                    mouths = get_mouth_pen_ratio( mouth_points, landmarks) 

                    left_eye_ratio = get_blinking_ratio( l_eye_poits, landmarks) 
                    right_eye_ratio = get_blinking_ratio( r_eye_points, landmarks) 
                    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2 

                    
                    print(mouths, blinking_ratio)
                    
                    # show the frame 
                    cv2.imshow("Frame", image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    #key = cv2.waitKey(0)
                    # if the `q` key was pressed, break from the loop 
                    #if key == ord("q"): break
                else:
                    print("no object")
    except:
        print(time.time()-start, count)


