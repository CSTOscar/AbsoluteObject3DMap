from object_detect import object_detector as object_detector_
from video_process import video_process

CKPT_FILE_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/model_files/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
LABEL_FILE_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/label_files/mscoco_label_map.pbtxt'
NUM_CLASS = 90

VIDEO_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/video_files/video0.mp4'
FRAME_DIR_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files'
STEP = 10
images = video_process.capture_frames_from_video(VIDEO_PATH, STEP)

object_detector = object_detector_.ObjectorDetector(CKPT_FILE_PATH, LABEL_FILE_PATH, NUM_CLASS)

TEST_IMAGE_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files/image0.jpg'

object_detection_record = object_detector.detect_object(images[0])

print(object_detection_record)
