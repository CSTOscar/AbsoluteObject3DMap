from frame import frame as frame_
from camera import camera as camera_
from video_process import video_process

VIDEO_PATH_L = '../../data/video_files/test_left.mp4'
VIDEO_PATH_R = '../../data/video_files/test_right.mp4'
FRAME_DIR_PATH = '../../data/image_files'
STEP = 10

imagesL = video_process.capture_frames_from_video(VIDEO_PATH_L, STEP)
imagesR = video_process.capture_frames_from_video(VIDEO_PATH_R, STEP)

R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
T = [[10], [10], [10]]
camera = camera_.Camera(86500, 86500, 0.035, 3264 / 2, 2448 / 2, R, T)

frames = frame_.generate_raw_frame_chain_from_images(imagesL, imagesR, camera)

for i in range(1, len(frames)):
    frame = frames[i]
    frame.generate_set_kp_des()

for i in range(1, len(frames)):
    frame = frames[i]
    frame.generate_set_motion_info()
