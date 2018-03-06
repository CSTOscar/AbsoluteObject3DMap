import cv2
import numpy as np


def capture_frames_from_video(video_path, step, frame_name_format='image{}'):
    video_capture = cv2.VideoCapture(video_path)
    count = 0
    # frame_file_path_format = os.path.join(frame_dir_path, frame_file_name_format)
    image_count = 0
    success = True
    images = []
    while success:
        # cv2.imwrite("cv_image/frames/image{}.jpg".format(count), image)  # save frame as JPEG file
        success, image = video_capture.read()
        count += 1
        if count % step == 0 and success:
            # cv2.imwrite(frame_file_path_format.format(image_count), image)
            images.append(image)
            image_count += 1
    return images
