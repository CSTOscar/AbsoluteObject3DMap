import numpy as np

from camera import camera
import frame as fm

IMAGE_NAME_FORMAT = 'image{}.jpg'

R = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
T = np.asmatrix([[0], [0], [0]])
camera1 = camera.Camera(86500, 86500, 0.035, 2448 / 2, 3264 / 2, R, T)

slam_info = np.load('../data/temp_files/results/image_SLAM_record.npy').item()[IMAGE_NAME_FORMAT.format(0)]
detection_info = np.load('../data/temp_files/results/image_detection_record.npy').item()[IMAGE_NAME_FORMAT.format(0)]

# print(slam_info)

print(slam_info)
print(detection_info)

print('test begin ---------------------')

frame_test = fm.Frame(detection_info, slam_info, camera1)

objects_inframe = frame_test.get_objects_with_confidence_more_than(0.0)

print(str(objects_inframe[1]))
