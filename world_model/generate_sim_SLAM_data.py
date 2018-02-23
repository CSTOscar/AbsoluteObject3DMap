import numpy as np

image_width = 3264
image_height = 2448

SLAM_results = {}

image_name_format = 'image{}.jpg'

for i in range(8):
    result = {}
    depth = np.random.normal(10, 20, [image_width, image_height])
    position = np.asmatrix([[5], [10], [15]])
    direction = np.asmatrix([[1, 0, 0]])
    print(depth)

    result['position'] = position
    result['direction'] = direction
    result['depth'] = depth

    SLAM_results[image_name_format.format(i)] = result

print('the shape of the depth map', SLAM_results['image5.jpg'])

np.save('../data/temp_files/results/image_SLAM_record', SLAM_results)
