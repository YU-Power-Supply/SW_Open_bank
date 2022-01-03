import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model


def CropFaceComponent(img, componentCoordinate, IMG_SIZE = (34, 26)):
  x1, y1 = np.amin(componentCoordinate, axis=0) # return lefttop coordinate
  x2, y2 = np.amax(componentCoordinate, axis=0) # return rightbottom coordinate
  cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2) #return senter coordinate by integer

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]
  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)
  
  coordinates = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)
  componentImage = img[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]

  return componentImage, coordinates


def main():

  trainedEyeSize = (34, 26)
  trainedMouthSize = (70, 50) # Temporary Value


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

      leftEyeImage, leftEyeCoordinate = CropFaceComponent(img = gray_img, componentCoordinate=shapes[36:42]) # Coordinate = 좌표 , 구성 = [[min x] [max x] [min y] [max y]]
      rightEyeImage, rightEyeCoordinate = CropFaceComponent(img = gray_img, componentCoordinate=shapes[42:48])
      mouthImage, mouthCoordinate = CropFaceComponent(img = gray_img, componentCoordinate=shapes[48:60])

      leftEyeImage = cv2.resize(leftEyeImage, dsize=trainedEyeSize)
      rightEyeImage = cv2.resize(rightEyeImage, dsize=trainedMouthSize)
      rightEyeImage = cv2.flip(rightEyeImage, flipCode=1) # 물어보기
      mouthImage = cv2.resize(mouthImage, dsize=trainedMouthSize)

      cv2.imshow('LeftEyeImage', leftEyeImage)
      cv2.imshow('RightEyeImage', rightEyeImage)
      cv2.imshow('MouthImage', mouthImage)

      putLeftEyeInModel = leftEyeImage.copy().reshape((1, trainedEyeSize[1], trainedEyeSize[0], 1)).astype(np.float32) / 255.
      putRightEyeInModel = rightEyeImage.copy().reshape((1, trainedEyeSize[1], trainedEyeSize[0], 1)).astype(np.float32) / 255.
      putMouthInModel = mouthImage.copy().reshape((1, trainedMouthSize[1], trainedMouthSize[0], 1)).astype(np.float32) / 255.

      predLeftEye = model.predict(putLeftEyeInModel)
      predRightEye = model.predict(putRightEyeInModel)
      predMouth = model.predict(putMouthInModel)
      

      # visualize
      stateLeftEye = f'O [{int(predLeftEye*100)}%]' if predLeftEye > 0.1 else f'- [{int(predLeftEye*100)}%]'
      stateRightEye = f'O [{int(predRightEye*100)}%]' if predRightEye > 0.1 else f'- [{int(predRightEye*100)}%]'
      stateMouth = f'O [{int(predRightEye*100)}%]' if predRightEye > 0.1 else f'- [{int(predRightEye*100)}%]'

      cv2.rectangle(img, pt1=tuple(leftEyeCoordinate[0:2]), pt2=tuple(leftEyeCoordinate[2:4]), color=(255,255,255), thickness=2)
      cv2.rectangle(img, pt1=tuple(rightEyeCoordinate[0:2]), pt2=tuple(rightEyeCoordinate[2:4]), color=(255,255,255), thickness=2)
      cv2.rectangle(img, pt1=tuple(mouthCoordinate[0:2]), pt2=tuple(mouthCoordinate[2:4]), color=(255,255,255), thickness=2)

      cv2.putText(img, stateLeftEye, tuple(leftEyeCoordinate[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
      cv2.putText(img, stateRightEye, tuple(rightEyeCoordinate[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
      cv2.putText(img, stateMouth, tuple(mouthCoordinate[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
      break


if __name__ =="__main__":
  main()