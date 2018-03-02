from frame import frame as frame_
from camera import camera as camera_
from video_process import video_process
import numpy as np

VIDEO_PATH_L = '../../data/video_files/test_left.mp4'
VIDEO_PATH_R = '../../data/video_files/test_right.mp4'
VIDEO_PATH_C = '/Users/zijunyan/Downloads/video_cal.mp4'
FRAME_DIR_PATH = '../../data/image_files'
STEP = 10

images_cal = video_process.capture_frames_from_video(VIDEO_PATH_C, 5)

imagesL = video_process.capture_frames_from_video(VIDEO_PATH_L, STEP)
imagesR = video_process.capture_frames_from_video(VIDEO_PATH_R, STEP)

R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
T = [[10], [10], [10]]
camera = camera_.Camera(86500, 86500, 0.035, 3264 / 2, 2448 / 2, R, T)

K_sum = None

num = 0

for image_cal in images_cal:
    updated = camera.calibrate_by_images_and_grid_length(image_cal, 0.024)
    if updated:
        num += 1
        print(camera.K)
        K_sum = camera.K if num == 1 else K_sum + camera.K

print('avr K')
K =
print(K_sum / num)

print(camera.K)

frames = frame_.generate_raw_frame_chain_from_images(imagesL, imagesR, camera)

for i in range(0, len(frames)):
    frame = frames[i]
    frame.generate_set_kp_des()

for i in range(1, len(frames)):
    frame = frames[i]
    frame.generate_set_motion_info()
