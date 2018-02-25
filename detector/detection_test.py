import object_detect.detection as od
import numpy as np

# What model
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
CKPT_FILE_NAME = 'frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
LABEL_FILE_NAME = 'mscoco_label_map.pbtxt'
NUM_CLASSES = 90

od.detection(MODEL_NAME, CKPT_FILE_NAME, LABEL_FILE_NAME, NUM_CLASSES, 'image{}.jpg', 8)

data = np.load('/Users/zijunyan/Desktop/Oscar/ObsoluteObject3DMap/data/temp_files/results/image_detection_record.npy')


