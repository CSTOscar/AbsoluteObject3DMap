from video_process import video_process

VIDEO_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/video_files/video0.mp4'
FRAME_DIR_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files'
STEP = 10
images = video_process.capture_frames_from_video(VIDEO_PATH, STEP)
print(images[0].shape)
