import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model


def crop_eye(img, eye_points, IMG_SIZE = (34, 26)):
  x1, y1 = np.amin(eye_points, axis=0) # return lefttop coordinate
  x2, y2 = np.amax(eye_points, axis=0) # return rightbottom coordinate
  cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2) #return senter coordinate by integer

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]
  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)
  
  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect


def main():

  IMG_SIZE = (34, 26)

  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

  model = load_model('models/2018_12_17_22_58_35.h5')
  model.summary()

  video_path = ''
  image_path = ''


  cap = cv2.VideoCapture(video_path)

  while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
      break

    frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)

    img = frame.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_img, 1) # The 1 in the second argumet indicates that we should upsample the image 1 time

    for face in faces:
      shapes = predictor(gray_img, face)
      shapes = face_utils.shape_to_np(shapes)

      leftEyeImage, leftEyeCoordinate = crop_eye(gray_img, eye_points=shapes[36:42]) # Coordinate = 좌표 , 구성 = [[min x] [max x] [min y] [max y]]
      rightEyeImage, rightEyeCoordinate = crop_eye(gray_img, eye_points=shapes[42:48])

      leftEyeImage = cv2.resize(leftEyeImage, dsize=IMG_SIZE)
      rightEyeImage = cv2.resize(rightEyeImage, dsize=IMG_SIZE)
      rightEyeImage = cv2.flip(rightEyeImage, flipCode=1)

      cv2.imshow('leftEyeImage', leftEyeImage)
      cv2.imshow('rightEyeImage', rightEyeImage)

      putLeftEye = leftEyeImage.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
      putRightEye = rightEyeImage.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

      pred_l = model.predict(putLeftEye)
      pred_r = model.predict(putRightEye)

      # visualize
      state_l = f'O [{int(pred_l*100)}%]' if pred_l > 0.1 else f'- [{int(pred_l*100)}%]'
      state_r = f'O [{int(pred_r*100)}%]' if pred_r > 0.1 else f'- [{int(pred_r*100)}%]'

      cv2.rectangle(img, pt1=tuple(leftEyeCoordinate[0:2]), pt2=tuple(leftEyeCoordinate[2:4]), color=(255,255,255), thickness=2)
      cv2.rectangle(img, pt1=tuple(rightEyeCoordinate[0:2]), pt2=tuple(rightEyeCoordinate[2:4]), color=(255,255,255), thickness=2)

      cv2.putText(img, state_l, tuple(leftEyeCoordinate[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
      cv2.putText(img, state_r, tuple(rightEyeCoordinate[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
      break


if __name__ =="__main__":
  main()