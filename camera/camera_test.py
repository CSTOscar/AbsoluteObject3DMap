import camera as camera
import numpy as np

# ground measure
# obj: 0.19  dist: 0.235 foc: 0.035 pix: 2448 mx: 86500
# this calculation is terrible, just for trail

# TODO: this is the most vulnerable, need more test

R = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
T = np.asmatrix([[0], [0], [0]])
camera1 = camera.Camera(86500, 86500, 0.035, 2448 / 2, 3264 / 2, R, T)

print('K,RT check')
print(camera1.K)
print(camera1.RT)

coordinate_world = np.asmatrix([[-0.095], [-0.095], [0.235], [1]])
coordinate_pixel = camera1.world_to_pixel(coordinate_world)
coordinate_pixel /= coordinate_pixel[2]

print(coordinate_pixel)

coordinate_world_back = camera1.pixel_to_world(coordinate_pixel)
# coordinate_world_back[3] = 0

coordinate_world_with_depth = camera1.pixel_depth_to_world(coordinate_pixel, 0.235)

print('coordinate_world_with_depth')
print(coordinate_world_with_depth)

print(coordinate_world_back)

coordinate_pixel_again = camera1.world_to_pixel(coordinate_world_back)

print(coordinate_pixel_again)

print(camera1.generate_camera_position_orientation_from_R_T()[1])
