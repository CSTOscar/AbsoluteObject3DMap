from frame import frame as frame_
from camera import camera as camera_
from video_process import video_process
import numpy as np
from detector.object_detector import ObjectorDetector
import cv2

# ffmpeg -i movie.mov -vcodec copy -acodec copy out.mp4

# TODO: move the file to fit the path
VIDEO_PATH_L = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/video_files/test_left.mp4'
VIDEO_PATH_R = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/video_files/test_right.mp4'
VIDEO_PATH_C = '/Users/zijunyan/Downloads/video_cal.mp4'
FRAME_DIR_PATH = '../../data/image_files'
STEP = 100

# images_cal = video_process.capture_frames_from_video(VIDEO_PATH_C, 5)

imagesL = video_process.capture_frames_from_video(VIDEO_PATH_L, STEP)
imagesR = video_process.capture_frames_from_video(VIDEO_PATH_R, STEP)

print(imagesL[0].shape)

R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
T = [[10], [10], [10]]
camera = camera_.Camera(86500, 86500, 0.035, 3264 / 2, 2448 / 2, R, T)

# K_sum = None
#
# num = 0
#
# for image_cal in images_cal:
#     updated = camera.calibrate_by_images_and_grid_length(image_cal, 0.024)
#     if updated:
#         num += 1
#         print(camera.K)
#         K_sum = camera.K if num == 1 else K_sum + camera.K
#
# print('avr K')
# avrK = K_sum / num
# print(avrK)

camera.K = np.asmatrix([[1.12265949e+03, 0.00000000e+00, 3.07322388e+02, 0.00000000e+00],
                        [0.00000000e+00, 1.11948374e+03, 5.50850887e+02, 0.00000000e+00],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])

camera.update_M_M_pinv_by_K_RT()

print(camera.K)
print(type(camera.K))
print(camera.M)

# for i in range(len(imagesL)):
#     imagesL[i] = np.transpose(imagesL[i], axes=(1, 0, 2))
#
# for i in range(len(imagesR)):
#     imagesR[i] = np.transpose(imagesR[i], axes=(1, 0, 2))
#     print(imagesR[i].shape)

frames = frame_.generate_raw_frame_chain_from_images(imagesL, imagesR, camera)

for i in range(0, len(frames)):
    frame = frames[i]
    frame.generate_set_kp_des()

print('generate kp and des done')

for i, frame in enumerate(frames):
    if i != 0:
        frame.generate_set_depth_info()
        # print(frame.depth_info)
        # print(frame.depth_info_generated)

print('generate depth is done')

for i in range(1, len(frames)):
    print('progress check for motion; ', i)
    frame = frames[i]
    frame.generate_set_motion_info()

frames[0].camera_extrinsic_set = True

for i in range(1, len(frames)):
    print('progress check for ex; ', i)
    frame = frames[i]
    frame.generate_update_camera_extrinsic_parameters_based_on_prev_frame()
    # position, direction = frame.camera.generate_camera_position_direction_from_R_T()
    # print(position, direction)

# for i, frame in enumerate(frames):
#     print(i, frame.motion_info_generated)
#     print(i, frame.camera_extrinsic_set)
#     print(i, frame.detection_info_generated)

CKPT_FILE_PATH = '../../data/model_files/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
LABEL_FILE_PATH = '../../data/label_files/mscoco_label_map.pbtxt'
NUM_CLASS = 90

objdet = ObjectorDetector(CKPT_FILE_PATH, LABEL_FILE_PATH, NUM_CLASS)

for i, frame in enumerate(frames):
    # print('object detection check: ', i, frames[i].detection_info_generated)
    frame.generate_set_detection_info(objdet)
    print('progress check for detection: ', i)
    print(frames[i].detection_info)

for i, frame in enumerate(frames):
    if i != 0:
        frame.generate_set_projections()

projections = frame_.collect_projections_from_frames(frames, confidence=0.70)
print(projections)
for frame in frames:
    position, direction = frame.camera.generate_camera_position_direction_from_R_T()
    print('fdsa', position, direction)

check_images = frame_.object_depth_detection_check_plot(frames[1:])

IMAGE_PATH_FORMAT = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files/check_image/image{}.jpg'
for i, image in enumerate(check_images):
    cv2.imwrite(IMAGE_PATH_FORMAT.format(i), image)

frame_.motion_check_plot(frames)
