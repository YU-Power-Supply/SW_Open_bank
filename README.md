# SW_Open_bank

## ClassificationSleepOrNoneSleep (최지웅, 김태환)
2018_12_17_22_58_35.h5 : faceDetection을 하기 위한 학습된 모델
ClassificationSleepOrNoneSleepByCnn.py : cnn 으로 학습망을 구성한 코드 (incomplete)
ClassificationSleepOrNoneSleepByDnn.py : dnn 으로 학습망을 구성한 코드 (complete)
shape_predictor_68_face_landmark.dat : detect landmark as keypoint on deteceted face 를 위한 학습된 모델


## darknet_setting.py
***
⇒ [[AI허브] 졸음운전 예방 데이터](http://aihub.or.kr/aidata/30744) 다음 데이터셋에 맞춰 제작된 프로그램입니다.

### setting values

1. directory : 다크넷 초기 세팅파일을(~.data ~.names .etc) 묶어서 저장할 디렉토리 명을 지정합니다.

2. img_dir : 이미지 디렉토리의 경로를 지정합니다.

3. json_dir : json 디렉토리의 경로를 지정합니다.

### darknet setting.py

* setnew

    처음 세팅을 할 때 사용합니다. 기존에 존재하는 세팅이 있다면 초기화됩니다.

* setadd

    기존 세팅에서 추가로 변경할 때 사용합니다. 

    (그 예시로 train.txt 또는 valid.txt 파일에 경로를 추가 등록할 때 사용합니다.)

* jtt

    주어진 데이터셋을 yolo 요구 데이터셋에 맞추기 위해 json 파일을 txt 파일로 변환합니다.

* rmv

    img_dir에 저장된 txt파일들을 지웁니다.

* kte

    yolo는 한글 파일명을 읽을 수 없기 때문에 영문 파일명으로 변경합니다.
