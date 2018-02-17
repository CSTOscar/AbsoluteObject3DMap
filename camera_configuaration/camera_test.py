import camera_configuaration.camera as camera1
import numpy as np

# ground measure
# obj: 0.19  dist: 0.235 foc: 0.035 pix: 2448 mx: 86500
# this calculation is terrible, just for trail

R = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
T = np.asmatrix([[1], [1], [1]])
camera1 = camera1.Camera(86500, 86500, 0.035, 2448 / 2, 3264 / 2, R, T)

coordinate_world = np.asmatrix([[-0.095], [-0.095], [0.235], [1]])
coordinate_pixel = camera1.world_to_pixel(coordinate_world)
coordinate_pixel /= coordinate_pixel[2]

print(coordinate_pixel)

coordinate_world_back = camera1.pixel_to_world(coordinate_pixel)
# coordinate_world_back[3] = 0

print(coordinate_world_back)

coordinate_pixel_again = camera1.world_to_pixel(coordinate_world_back)

print(coordinate_pixel_again)

print(camera1.generate_camera_position_orientation_from_R_T()[0])
