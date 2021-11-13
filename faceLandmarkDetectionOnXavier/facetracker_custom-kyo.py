import copy
import os
import sys
import traceback
import gc
from math import hypot
import matplotlib.pyplot as plt


mouth_points = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
r_eye_points = [42, 43, 44, 45, 46, 47]
l_eye_points = [36, 37, 38, 39, 40, 41]

def midpoint(p1, p2): 
    return int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2) 

def get_face_angle(frame, facial_landmarks): 

    left_point = (facial_landmarks[36*2], facial_landmarks[36*2+1])
    right_point = (facial_landmarks[45*2], facial_landmarks[45*2+1])
    
    left_toppoint = (facial_landmarks[0*2], facial_landmarks[0*2+1])
    right_toppoint = (facial_landmarks[16*2], facial_landmarks[16*2+1])
    
    left_eyebrow = (facial_landmarks[17*2], facial_landmarks[17*2+1])
    right_eyebrow = (facial_landmarks[26*2], facial_landmarks[26*2+1])
    
    left_len = left_point[0] - left_toppoint[0] #왼쪽눈끝점 - 왼쪽얼굴끝점
    right_len = right_toppoint[0] - right_point[0] #오른족 얼굴끝점- 오른쪽눈끝점
    
    #시연용 코드
    cv2.line(frame, left_eyebrow, right_eyebrow, (255, 0, 0), 2)
    cv2.line(frame, left_toppoint, right_toppoint, (0, 255, 0), 2)
    cv2.circle(frame, left_point, 5, (255, 0, 0), 2)
    cv2.circle(frame, right_point, 5, (255, 0, 0), 2)
    cv2.circle(frame, left_toppoint, 5, (0, 255, 0), 2)
    cv2.circle(frame, right_toppoint, 5, (0, 255, 0), 2)
    
    
    if left_len < 0 or right_len < 0:
        cv2.putText(frame,"yaw over!!",org,font,1,(255,0,255),2)
        return True
    if abs(right_point[1] - left_point[1]) / (right_point[0]-left_point[0]) > 0.176: # tan(10도) = 0.176 => 10도이상 넘어가면 감지
        cv2.putText(frame,"roll over!!",org,font,1,(255,255,0),2)
        return True
    if midpoint(left_eyebrow, right_eyebrow)[1] > midpoint(left_toppoint, right_toppoint)[1]:
        cv2.putText(frame,"pitch over!!",org,font,1,(0,255,255),2)
        return True
    
    return False

def get_blinking_ratio(frame, eye_points, facial_landmarks): 
    center_top = midpoint(facial_landmarks[eye_points[1]*2:eye_points[1]*2+2], facial_landmarks[eye_points[2]*2:eye_points[2]*2+2])
    center_bottom = midpoint(facial_landmarks[eye_points[4]*2:eye_points[4]*2+2], facial_landmarks[eye_points[5]*2:eye_points[5]*2+2])
    
    left_point = (facial_landmarks[eye_points[0]*2], facial_landmarks[eye_points[0]*2+1])
    right_point = (facial_landmarks[eye_points[3]*2], facial_landmarks[eye_points[3]*2+1])

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2) 
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2) 
    hor_line_lenght = hypot( (left_point[0] - right_point[0]), (left_point[1] - right_point[1])) 
    ver_line_lenght = hypot( (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1])) 
    ratio = hor_line_lenght / ver_line_lenght
    if ver_line_lenght != 0: 
        ratio = ver_line_lenght / hor_line_lenght
    else: 
        ratio = 60 
    return ratio 

def eye_checker(frame, l_eye_points, r_eye_points, facial_landmarks, sleep_check):
    left_eye_ratio = get_blinking_ratio(frame, l_eye_points, facial_landmarks)
    right_eye_ratio = get_blinking_ratio(frame, r_eye_points, facial_landmarks)
    
    if (left_eye_ratio + right_eye_ratio) / 2 < 0.18:
        return sleep_check+1, (left_eye_ratio + right_eye_ratio) / 2
    else:
        return 0 , (left_eye_ratio + right_eye_ratio) / 2
    

max_threads = 1
os.environ["OMP_NUM_THREADS"] = str(max_threads)

import numpy as np
import time
import cv2
from input_reader import InputReader, DShowCaptureReader, try_int
from trackerkyo import Tracker
org=(50,75) 
font=cv2.FONT_HERSHEY_SIMPLEX

def run(fps=15, visualize = 0, dcap=None, use_dshowcapture=1, capture="0",raw_rgb=0, width=640, height=360, video_out = None, face_id_offset = 0, video_scale=1, threshold=None, max_threads=max_threads, faces=1, discard_after=10, scan_every=3, silent=0, model=3, model_dir=None, gaze_tracking=1, detection_threshold=0.6, scan_retinaface=0, max_feature_updates=900, no_3d_adapt=1, try_hard=0):
    
    use_dshowcapture_flag = False
    if os.name == 'nt':
        use_dshowcapture_flag = True if use_dshowcapture == 1 else False
        input_reader = InputReader(capture, raw_rgb, width, height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
        if dcap == -1 and type(input_reader) == DShowCaptureReader:
            fps = min(fps, input_reader.device.get_fps())
    else:
        input_reader = InputReader(capture, raw_rgb, width, height, fps, use_dshowcapture=use_dshowcapture_flag)
        
    out = None
    first = True
    fheight = 0
    fwidth = 0
    tracker = None
    sock = None
    total_tracking_time = 0.0
    tracking_time = 0.0
    tracking_frames = 0
    frame_count = 0
    sleep_check = 0
    plotdata = [[], []]
    plotx = 0

    features = ["eye_l", "eye_r", "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]

    is_camera = capture == str(try_int(capture))

    try:
        attempt = 0
        frame_time = time.perf_counter()
        target_duration = 0
        if fps > 0:
            target_duration = 1. / float(fps)
        need_reinit = 0
        failures = 0
        source_name = input_reader.name
        A_frame = np.empty((0, 136), dtype=int)
        while input_reader.is_open():
            if not input_reader.is_open() or need_reinit == 1:
                input_reader = InputReader(capture, raw_rgb, width, height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
                if input_reader.name != source_name:
                    print(f"Failed to reinitialize camera and got {input_reader.name} instead of {source_name}.")
                    sys.exit(1)
                need_reinit = 2
                time.sleep(0.02)
                continue
            if not input_reader.is_ready():
                time.sleep(0.02)
                continue
            ret, frame = input_reader.read()
            if not ret:
                if is_camera:
                    attempt += 1
                    if attempt > 30:
                        break
                    else:
                        time.sleep(0.02)
                        if attempt == 3:
                            need_reinit = 1
                        continue
                else:
                    break;

            attempt = 0
            need_reinit = 0
            frame_count += 1
            if first:
                first = False
                fheight, fwidth, _ = frame.shape
                tracker = Tracker(fwidth, fheight, threshold=threshold, max_threads=max_threads, max_faces=faces, discard_after=discard_after, scan_every=scan_every, silent=False if silent == 0 else True, model_type=model, model_dir=model_dir, no_gaze=False if gaze_tracking != 0 and model != -1 else True, detection_threshold=detection_threshold, use_retinaface=scan_retinaface, max_feature_updates=max_feature_updates, static_model=True if no_3d_adapt == 1 else False, try_hard=try_hard == 1)
                if not video_out is None:
                    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc('F','F','V','1'), fps, (fwidth * video_scale, fheight * video_scale))

            try:
                inference_start = time.perf_counter()
                faces = tracker.predict(frame)
                if len(faces) > 0:
                    inference_time = (time.perf_counter() - inference_start)
                    total_tracking_time += inference_time
                    tracking_time += inference_time / len(faces)
                    tracking_frames += 1

                landmarks = np.array([], int) # landmarks in a frame
                for _, f in enumerate(faces):
                    f = copy.copy(f)
                    f.id += face_id_offset

                    for pt_num, (x,y,c) in enumerate(f.lms):
                        x = int(x + 0.5)
                        y = int(y + 0.5)
                        
                        landmarks = np.append(landmarks, [y, x], axis=0)
                        if visualize != 0 or not out is None:
                            color = (0, 255, 0)
                            if pt_num >= 66:
                                color = (255, 255, 0)
                            if not (x < 0 or y < 0 or x >= fheight or y >= fwidth):
                                frame[int(x), int(y)] = color
                            x += 1
                            if not (x < 0 or y < 0 or x >= fheight or y >= fwidth):
                                frame[int(x), int(y)] = color
                            y += 1
                            if not (x < 0 or y < 0 or x >= fheight or y >= fwidth):
                                frame[int(x), int(y)] = color
                            x -= 1
                            if not (x < 0 or y < 0 or x >= fheight or y >= fwidth):
                                frame[int(x), int(y)] = color
                                
                    if f.current_features is None:
                        f.current_features = {}
                    for feature in features:
                        if not feature in f.current_features:
                            f.current_features[feature] = 0
                
                if get_face_angle(frame, landmarks):
                    head_check = fps
                else:
                    head_check = fps * 2
                plotx += 1
                sleep_check, ploty = eye_checker(frame, l_eye_points, r_eye_points, landmarks, sleep_check)
                plotdata[0].append(plotx)
                plotdata[1].append(ploty)
                
                
                if sleep_check > head_check:
                    cv2.putText(frame,"Wake up!!",(50, 50),font,1,(255,0,0),2)
                
                if landmarks.size != 136:
                    landmarks = np.append(landmarks, np.zeros(136-landmarks.size), axis=0)
                A_frame = np.vstack([A_frame, landmarks])

                if not out is None:
                    video_frame = frame
                    if video_scale != 1:
                        video_frame = cv2.resize(frame, (fwidth * video_scale, fheight * video_scale), interpolation=cv2.INTER_NEAREST)
                    out.write(video_frame)
                    if video_scale != 1:
                        del video_frame

                if visualize != 0:
                    cv2.imshow('OpenSeeFace Visualization', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                failures = 0

            except Exception as e:
                if e.__class__ == KeyboardInterrupt:
                    if silent == 0:
                        print("Quitting")
                    break
                traceback.print_exc()
                failures += 1
                if failures > 30:
                    break

            collected = False
            del frame

            duration = time.perf_counter() - frame_time
            while duration < target_duration:
                if not collected:
                    gc.collect()
                    collected = True
                duration = time.perf_counter() - frame_time
                sleep_time = target_duration - duration
                if sleep_time > 0:
                    time.sleep(sleep_time)
                duration = time.perf_counter() - frame_time
            frame_time = time.perf_counter()
    except KeyboardInterrupt:
        if silent == 0:
            print("Quitting")

    input_reader.close()
    if not out is None:
        out.release()
    cv2.destroyAllWindows()

    plt.plot(plotdata[0], plotdata[1], 'ro', plotdata[0], plotdata[1], 'b--')
    plt.show()

    if silent == 0 and tracking_frames > 0:
        average_tracking_time = 1000 * tracking_time / tracking_frames
        print(f"Average tracking time per detected face: {average_tracking_time:.2f} ms")
        print(f"Tracking time: {total_tracking_time:.3f} s\nFrames: {tracking_frames}\nFPS: {tracking_frames/total_tracking_time:.3f}")
    return A_frame
if __name__ == "__main__":
    frame = run(visualize=1, max_threads=4, capture="video1.mp4")
    print(frame, frame.size)
    plt.show()