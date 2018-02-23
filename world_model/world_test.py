import world_model.world as world_
from camera import camera as camera_
from frame import frame as frame_
import numpy as np

world = world_.World()

IMAGE_NAME_FORMAT = 'image{}.jpg'

R = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
T = np.asmatrix([[0], [0], [0]])

default_camera_parameters = (86500, 86500, 0.035, 2448 / 2, 3264 / 2, R, T)

slam_info = np.load('../data/temp_files/results/image_SLAM_record.npy').item()
detection_info = np.load('../data/temp_files/results/image_detection_record.npy').item()

# print(slam_info.keys())
# print(detection_info.keys())

for key in slam_info.keys():
    camera = camera_.Camera(*default_camera_parameters)
    print(key)
    frame = frame_.Frame(detection_info[key], slam_info[key], camera)
    world.add_objects_projection_from_frame(frame, 0.5)

# for item in world.objects_projection:
#     print(item)

clazz = set([])
for projection in world.objects_projection:
    clazz.add(projection['class'])

print(clazz)

print('object unify begins ------------------------------')
world.unify_objects_projection_get_object()

for obj in world.objects:
    print(obj['class'], obj['position'])
