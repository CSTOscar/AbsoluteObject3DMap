from detector import object_detector as object_detector_
from video_process import video_process
import cv2
import numpy as np
from utils import visualization_utils as vis_util
from matplotlib import pyplot as plt

CKPT_FILE_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/model_files/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
LABEL_FILE_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/label_files/mscoco_label_map.pbtxt'
NUM_CLASS = 90

VIDEO_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/video_files/video0.mp4'
FRAME_DIR_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files'
STEP = 10

# images = video_process.capture_frames_from_video(VIDEO_PATH, STEP)

image = cv2.imread('/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files/image_l/MouldShotTest1_l.JPG')

object_detector = object_detector_.ObjectorDetector(CKPT_FILE_PATH, LABEL_FILE_PATH, NUM_CLASS)

TEST_IMAGE_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files/image0.jpg'

object_detection_record, (boxes, scores, classes, num) = object_detector.detect_object(image)

print(object_detection_record)

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    object_detector.category_index,
    use_normalized_coordinates=True,
    line_thickness=8)
plt.figure()
plt.imshow(image)

image = cv2.imread('/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files/image_l/MouldShotTest2_l.JPG')

object_detector = object_detector_.ObjectorDetector(CKPT_FILE_PATH, LABEL_FILE_PATH, NUM_CLASS)

TEST_IMAGE_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files/image0.jpg'

object_detection_record, (boxes, scores, classes, num) = object_detector.detect_object(image)

print(object_detection_record)

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    object_detector.category_index,
    use_normalized_coordinates=True,
    line_thickness=8)
plt.figure()
plt.imshow(image)

image = cv2.imread('/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files/image_l/MouldShotTest3_l.JPG')

object_detector = object_detector_.ObjectorDetector(CKPT_FILE_PATH, LABEL_FILE_PATH, NUM_CLASS)

TEST_IMAGE_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files/image0.jpg'

object_detection_record, (boxes, scores, classes, num) = object_detector.detect_object(image)

print(object_detection_record)

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    object_detector.category_index,
    use_normalized_coordinates=True,
    line_thickness=8)
plt.figure()
plt.imshow(image)

plt.show()
