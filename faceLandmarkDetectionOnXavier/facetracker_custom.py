import copy
import os
import sys
import argparse
import traceback
import gc
import dshowcapture

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--ip", help="Set IP address for sending tracking data", default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, help="Set port for sending tracking data", default=11573)
if os.name == 'nt':
    parser.add_argument("-l", "--list-cameras", type=int, help="Set this to 1 to list the available cameras and quit, set this to 2 or higher to output only the names", default=0)
    parser.add_argument("-a", "--list-dcaps", type=int, help="Set this to -1 to list all cameras and their available capabilities, set this to a camera id to list that camera's capabilities", default=None)
    parser.add_argument("-W", "--width", type=int, help="Set camera and raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set camera and raw RGB height", default=360)
    parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=24)
    parser.add_argument("-D", "--dcap", type=int, help="Set which device capability line to use or -1 to use the default camera settings", default=None)
    parser.add_argument("-B", "--blackmagic", type=int, help="When set to 1, special support for Blackmagic devices is enabled", default=0)
else:
    parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=360)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-M", "--mirror-input", action="store_true", help="Process a mirror image of the input video")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking, to 2 to also show face ids, to 3 to add confidence values or to 4 to add numbers to the point display", default=0)
parser.add_argument("-P", "--pnp-points", type=int, help="Set this to 1 to add the 3D fitting points to the visualization", default=0)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=0)
parser.add_argument("--faces", type=int, help="Set the maximum number of faces (slow)", default=1)
parser.add_argument("--scan-retinaface", type=int, help="When set to 1, scanning for additional faces will be performed using RetinaFace in a background thread, otherwise a simpler, faster face detection mechanism is used. When the maximum number of faces is 1, this option does nothing.", default=0)
parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=3)
parser.add_argument("--discard-after", type=int, help="Set the how long the tracker should keep looking for lost faces", default=10)
parser.add_argument("--max-feature-updates", type=int, help="This is the number of seconds after which feature min/max/medium values will no longer be updated once a face has been detected.", default=900)
parser.add_argument("--no-3d-adapt", type=int, help="When set to 1, the 3D face model will not be adapted to increase the fit", default=1)
parser.add_argument("--try-hard", type=int, help="When set to 1, the tracker will try harder to find a face", default=0)
parser.add_argument("--video-out", help="Set this to the filename of an AVI file to save the tracking visualization as a video", default=None)
parser.add_argument("--video-scale", type=int, help="This is a resolution scale factor applied to the saved AVI file", default=1, choices=[1,2,3,4])
parser.add_argument("--video-fps", type=float, help="This sets the frame rate of the output AVI file", default=24)
parser.add_argument("--raw-rgb", type=int, help="When this is set, raw RGB frames of the size given with \"-W\" and \"-H\" are read from standard input instead of reading a video", default=0)
parser.add_argument("--log-data", help="You can set a filename to which tracking data will be logged here", default="")
parser.add_argument("--log-output", help="You can set a filename to console output will be logged here", default="")
parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed, except for model 4, which is wink optimized. Models 1 and 0 tend to be too rigid for expression and blink detection. Model -2 is roughly equivalent to model 1, but faster. Model -3 is between models 0 and -1.", default=3, choices=[-3, -2, -1, 0, 1, 2, 3, 4])
parser.add_argument("--model-dir", help="This can be used to specify the path to the directory containing the .onnx model files", default=None)
parser.add_argument("--gaze-tracking", type=int, help="When set to 1, experimental blink detection and gaze tracking are enabled, which makes things slightly slower", default=1)
parser.add_argument("--face-id-offset", type=int, help="When set, this offset is added to all face ids, which can be useful for mixing tracking data from multiple network sources", default=0)
parser.add_argument("--repeat-video", type=int, help="When set to 1 and a video file was specified with -c, the tracker will loop the video until interrupted", default=0)
parser.add_argument("--dump-points", type=str, help="When set to a filename, the current face 3D points are made symmetric and dumped to the given file when quitting the visualization with the \"q\" key", default="")
parser.add_argument("--benchmark", type=int, help="When set to 1, the different tracking models are benchmarked, starting with the best and ending with the fastest and with gaze tracking disabled for models with negative IDs", default=0)
if os.name == 'nt':
    parser.add_argument("--use-dshowcapture", type=int, help="When set to 1, libdshowcapture will be used for video input instead of OpenCV", default=1)
    parser.add_argument("--blackmagic-options", type=str, help="When set, this additional option string is passed to the blackmagic capture library", default=None)
    parser.add_argument("--priority", type=int, help="When set, the process priority will be changed", default=None, choices=[0, 1, 2, 3, 4, 5])

max_threads = 1
os.environ["OMP_NUM_THREADS"] = str(max_threads)


def search_camera(list_cameras = 0, list_dcaps = -1):
    if os.name == 'nt' and (list_cameras > 0 or not list_dcaps is None):
        cap = dshowcapture.DShowCapture()
        info = cap.get_info()
        unit = 10000000.;
        if not list_dcaps is None:
            formats = {0: "Any", 1: "Unknown", 100: "ARGB", 101: "XRGB", 200: "I420", 201: "NV12", 202: "YV12", 203: "Y800", 300: "YVYU", 301: "YUY2", 302: "UYVY", 303: "HDYC (Unsupported)", 400: "MJPEG", 401: "H264" }
            for cam in info:
                if list_dcaps == -1:
                    type = ""
                    if cam['type'] == "Blackmagic":
                        type = "Blackmagic: "
                    print(f"{cam['index']}: {type}{cam['name']}")
                if list_dcaps != -1 and list_dcaps != cam['index']:
                    continue
                for caps in cam['caps']:
                    format = caps['format']
                    if caps['format'] in formats:
                        format = formats[caps['format']]
                    if caps['minCX'] == caps['maxCX'] and caps['minCY'] == caps['maxCY']:
                        print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
                    else:
                        print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']}-{caps['maxCX']}x{caps['maxCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
        else:
            if list_cameras == 1:
                print("Available cameras:")
            for cam in info:
                type = ""
                if cam['type'] == "Blackmagic":
                    type = "Blackmagic: "
                if list_cameras == 1:
                    print(f"{cam['index']}: {type}{cam['name']}")
                else:
                    print(f"{type}{cam['name']}")
        cap.destroy_capture()
    else:
        print("윈도우가 아니라서 실행할 수 없습니다.")

import numpy as np
import time
import cv2
import socket
import struct
import json
from input_reader import InputReader, VideoReader, DShowCaptureReader, try_int
from tracker import Tracker, get_model_base_path


def run(ip="127.0.0.1", port=11573, fps=24, visualize = 0, dcap=None, use_dshowcapture=1, capture="0", log_data="",raw_rgb=0, width=640, height=360, video_out = None, face_id_offset = 0, video_scale=1, threshold=None, max_threads=max_threads, faces=1, discard_after=10, scan_every=3, silent=0, model=3, model_dir=None, gaze_tracking=1, detection_threshold=0.6, scan_retinaface=0, max_feature_updates=900, no_3d_adapt=1, try_hard=0, video_fps = 24, dump_points = ""):
    
    use_dshowcapture_flag = False
    if os.name == 'nt':
        use_dshowcapture_flag = True if use_dshowcapture == 1 else False
        input_reader = InputReader(capture, raw_rgb, width, height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
        if dcap == -1 and type(input_reader) == DShowCaptureReader:
            fps = min(fps, input_reader.device.get_fps())
    else:
        input_reader = InputReader(capture, raw_rgb, width, height, fps, use_dshowcapture=use_dshowcapture_flag)
    if type(input_reader.reader) == VideoReader:
        fps = 0
        
    log = None
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

    features = ["eye_l", "eye_r", "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]

    if log_data != "":
        log = open(log_data, "w")
        log.write("Frame,Time,Width,Height,FPS,Face,FaceID,RightOpen,LeftOpen,AverageConfidence,Success3D,PnPError,RotationQuat.X,RotationQuat.Y,RotationQuat.Z,RotationQuat.W,Euler.X,Euler.Y,Euler.Z,RVec.X,RVec.Y,RVec.Z,TVec.X,TVec.Y,TVec.Z")
        for i in range(66):
            log.write(f",Landmark[{i}].X,Landmark[{i}].Y,Landmark[{i}].Confidence")
        for i in range(66):
            log.write(f",Point3D[{i}].X,Point3D[{i}].Y,Point3D[{i}].Z")
        for feature in features:
            log.write(f",{feature}")
        log.write("\r\n")
        log.flush()

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
            now = time.time()

            if first:
                first = False
                fheight, fwidth, channels = frame.shape
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                tracker = Tracker(fwidth, fheight, threshold=threshold, max_threads=max_threads, max_faces=faces, discard_after=discard_after, scan_every=scan_every, silent=False if silent == 0 else True, model_type=model, model_dir=model_dir, no_gaze=False if gaze_tracking != 0 and model != -1 else True, detection_threshold=detection_threshold, use_retinaface=scan_retinaface, max_feature_updates=max_feature_updates, static_model=True if no_3d_adapt == 1 else False, try_hard=try_hard == 1)
                if not video_out is None:
                    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc('F','F','V','1'), video_fps, (fwidth * video_scale, fheight * video_scale))

            try:
                inference_start = time.perf_counter()
                faces = tracker.predict(frame)
                if len(faces) > 0:
                    inference_time = (time.perf_counter() - inference_start)
                    total_tracking_time += inference_time
                    tracking_time += inference_time / len(faces)
                    tracking_frames += 1
                packet = bytearray()
                detected = False
                landmarks = np.empty((1, 2), dtype=float) # landmarks in a frame
                for face_num, f in enumerate(faces):
                    f = copy.copy(f)
                    f.id += face_id_offset
                    if f.eye_blink is None:
                        f.eye_blink = [1, 1]
                    right_state = "O" if f.eye_blink[0] > 0.30 else "-"
                    left_state = "O" if f.eye_blink[1] > 0.30 else "-"
                    if silent == 0:
                        print(f"Confidence[{f.id}]: {f.conf:.4f} / 3D fitting error: {f.pnp_error:.4f} / Eyes: {left_state}, {right_state}")
                    detected = True
                    if not f.success:
                        pts_3d = np.zeros((70, 3), np.float32)
                    packet.extend(bytearray(struct.pack("d", now)))
                    packet.extend(bytearray(struct.pack("i", f.id)))
                    packet.extend(bytearray(struct.pack("f", fwidth)))
                    packet.extend(bytearray(struct.pack("f", fheight)))
                    packet.extend(bytearray(struct.pack("f", f.eye_blink[0])))
                    packet.extend(bytearray(struct.pack("f", f.eye_blink[1])))
                    packet.extend(bytearray(struct.pack("B", 1 if f.success else 0)))
                    packet.extend(bytearray(struct.pack("f", f.pnp_error)))
                    packet.extend(bytearray(struct.pack("f", f.quaternion[0])))
                    packet.extend(bytearray(struct.pack("f", f.quaternion[1])))
                    packet.extend(bytearray(struct.pack("f", f.quaternion[2])))
                    packet.extend(bytearray(struct.pack("f", f.quaternion[3])))
                    packet.extend(bytearray(struct.pack("f", f.euler[0])))
                    packet.extend(bytearray(struct.pack("f", f.euler[1])))
                    packet.extend(bytearray(struct.pack("f", f.euler[2])))
                    packet.extend(bytearray(struct.pack("f", f.translation[0])))
                    packet.extend(bytearray(struct.pack("f", f.translation[1])))
                    packet.extend(bytearray(struct.pack("f", f.translation[2])))
                    if not log is None:
                        log.write(f"{frame_count},{now},{fwidth},{fheight},{fps},{face_num},{f.id},{f.eye_blink[0]},{f.eye_blink[1]},{f.conf},{f.success},{f.pnp_error},{f.quaternion[0]},{f.quaternion[1]},{f.quaternion[2]},{f.quaternion[3]},{f.euler[0]},{f.euler[1]},{f.euler[2]},{f.rotation[0]},{f.rotation[1]},{f.rotation[2]},{f.translation[0]},{f.translation[1]},{f.translation[2]}")
                    for (x,y,c) in f.lms:
                        packet.extend(bytearray(struct.pack("f", c)))
                    for pt_num, (x,y,c) in enumerate(f.lms):
                        packet.extend(bytearray(struct.pack("f", y)))
                        packet.extend(bytearray(struct.pack("f", x)))
                        landmarks = np.append(landmarks, np.array([[x, y]]), axis=0)
                        if not log is None:
                            log.write(f",{y},{x},{c}")
                        if pt_num == 66 and (f.eye_blink[0] < 0.30 or c < 0.20):
                            continue
                        if pt_num == 67 and (f.eye_blink[1] < 0.30 or c < 0.20):
                            continue
                        x = int(x + 0.5)
                        y = int(y + 0.5)
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
                                
                                
                    for (x,y,z) in f.pts_3d:
                        packet.extend(bytearray(struct.pack("f", x)))
                        packet.extend(bytearray(struct.pack("f", -y)))
                        packet.extend(bytearray(struct.pack("f", -z)))
                        if not log is None:
                            log.write(f",{x},{-y},{-z}")
                    if f.current_features is None:
                        f.current_features = {}
                    for feature in features:
                        if not feature in f.current_features:
                            f.current_features[feature] = 0
                        packet.extend(bytearray(struct.pack("f", f.current_features[feature])))
                        if not log is None:
                            log.write(f",{f.current_features[feature]}")
                    if not log is None:
                        log.write("\r\n")
                        log.flush()

                if detected and len(faces) < 40:
                    sock.sendto(packet, (ip, port))

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
                        if dump_points != "" and not faces is None and len(faces) > 0:
                            np.set_printoptions(threshold=sys.maxsize, precision=15)
                            pairs = [
                                (0, 16),
                                (1, 15),
                                (2, 14),
                                (3, 13),
                                (4, 12),
                                (5, 11),
                                (6, 10),
                                (7, 9),
                                (17, 26),
                                (18, 25),
                                (19, 24),
                                (20, 23),
                                (21, 22),
                                (31, 35),
                                (32, 34),
                                (36, 45),
                                (37, 44),
                                (38, 43),
                                (39, 42),
                                (40, 47),
                                (41, 46),
                                (48, 52),
                                (49, 51),
                                (56, 54),
                                (57, 53),
                                (58, 62),
                                (59, 61),
                                (65, 63)
                            ]
                            points = copy.copy(faces[0].face_3d)
                            for a, b in pairs:
                                x = (points[a, 0] - points[b, 0]) / 2.0
                                y = (points[a, 1] + points[b, 1]) / 2.0
                                z = (points[a, 2] + points[b, 2]) / 2.0
                                points[a, 0] = x
                                points[b, 0] = -x
                                points[[a, b], 1] = y
                                points[[a, b], 2] = z
                            points[[8, 27, 28, 29, 33, 50, 55, 60, 64], 0] = 0.0
                            points[30, :] = 0.0
                            with open(dump_points, "w") as fh:
                                fh.write(repr(points))
                        break
                failures = 0
                landmarks = np.delete(landmarks, [0, 0], axis=0)
                print(landmarks)
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

    if silent == 0 and tracking_frames > 0:
        average_tracking_time = 1000 * tracking_time / tracking_frames
        print(f"Average tracking time per detected face: {average_tracking_time:.2f} ms")
        print(f"Tracking time: {total_tracking_time:.3f} s\nFrames: {tracking_frames}\nFPS: {tracking_frames/total_tracking_time:.3f}")

if __name__ == "__main__":
    run(visualize=1, max_threads=4, capture="video.mp4")