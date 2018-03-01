from frame import frame as frame_
from camera import camera as camera_
from video_process import video_process

VIDEO_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/video_files/video0.mp4'
FRAME_DIR_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files'
STEP = 10

imagesL = video_process.capture_frames_from_video(VIDEO_PATH, STEP)
imagesR = video_process.capture_frames_from_video(VIDEO_PATH, STEP)

R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
T = [[10], [10], [10]]
camera = camera_.Camera(86500, 86500, 0.035, 3264 / 2, 2448 / 2, R, T)

frame_.generate_raw_frame_chain_from_images(imagesL, imagesR, camera)
