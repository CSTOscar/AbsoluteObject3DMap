from camera import camera as camera_
import numpy as np


def generate_R_inv_from_direction_test(D):
    print('---TEST---def generate_R_inv_from_direction(D)---BEGIN---')
    Z = np.asmatrix([[0.0], [0.0], [1.0]])
    test_case = [np.asmatrix(np.random.normal(1, 1, (3, 1))) for _ in range(10)]
    print('test case')
    print(test_case)

    test_case.append(-Z)
    R_invs = []
    validation_ground_truth = []
    for A in test_case:
        R_invs.append(camera_.Camera.generate_rotation_from_direction(A))
        validation_ground_truth.append(A / np.linalg.norm(A))
    validation = [R_invs[i] @ Z - validation_ground_truth[i] for i in range(len(R_invs))]
    print(validation)
    print('---TEST---def generate_R_inv_from_direction(D)---END---')


def camera_test():
    print('---TEST---Camera---BEGIN---')
    direction = [1, 0, 0]
    position = [1, 1, 1]
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T = np.asmatrix([[10], [10], [10]])
    camera = camera_.Camera(86500, 86500, 0.035, 3264 / 2, 2448 / 2, R, T)
    camera.update_extrinsic_parameters_by_camera_position_direction(position, direction)
    # camera.update_extrinsic_parameters_by_camera_position_rotation([0, 0, 0], R)
    depth = 1
    pixel_coordinate = [1632 * 2, 1224]
    world_coordinate = camera.pixel_depth_to_world(pixel_coordinate, depth)
    print(world_coordinate)
    world_coordinate2 = camera.pixel_to_world(pixel_coordinate)
    print(world_coordinate2)
    pixel_coordinate_back = camera.world_to_pixel(world_coordinate2)
    print(pixel_coordinate_back)
    pixel_coordinate_back = camera.world_to_pixel(world_coordinate)
    print(pixel_coordinate_back)

    position, direction = camera.generate_camera_position_direction_from_R_T()
    print(position, direction)
    print(camera.generate_camera_position_rotation_from_R_T())
    print(camera.R)
    print(camera.T)
    print('---TEST---Camera---END---')

    print(camera.pixel_to_world.original(camera, np.asmatrix([[10], [10], [10]])))

def camera_test2():
    print('---TEST---Camera---BEGIN---')
    direction = [0, 0, 1]
    position = [0, 0, 0]
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T = np.asmatrix([[10], [10], [10]])
    camera = camera_.Camera(86500, 86500, 0.035, 3264 / 2, 2448 / 2, R, T)
    camera.update_extrinsic_parameters_by_camera_position_direction(position, direction)
    # camera.update_extrinsic_parameters_by_camera_position_rotation([0, 0, 0], R)
    depth = 1
    pixel_coordinate = [1632, 1224]
    world_coordinate = camera.pixel_depth_to_world(pixel_coordinate, depth)
    print(world_coordinate)
    world_coordinate2 = camera.pixel_to_world(pixel_coordinate)
    print(world_coordinate2)
    pixel_coordinate_back = camera.world_to_pixel(world_coordinate2)
    print(pixel_coordinate_back)
    pixel_coordinate_back = camera.world_to_pixel(world_coordinate)
    print(pixel_coordinate_back)

    position, direction = camera.generate_camera_position_direction_from_R_T()
    print(position, direction)
    print(camera.generate_camera_position_rotation_from_R_T())
    print(camera.R)
    print(camera.T)
    print('---TEST---Camera---END---')

    print(camera.pixel_to_world.original(camera, np.asmatrix([[10], [10], [10]])))

# ground measure
# obj: 0.19  dist: 0.235 foc: 0.035 pix: 2448 mx: 86500
# this calculation is terrible, just for trail

# TODO: this is the most vulnerable, need more test

# R = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# T = np.asmatrix([[10], [10], [10]])
# camera1 = camera_.Camera(86500, 86500, 0.035, 2448 / 2, 3264 / 2, R, T)
#
# print('K,RT check')
# print(camera1.K)
# print(camera1.RT)
#
# coordinate_world = np.asmatrix([[-0.095], [0.095], [0.235], [1]])
# coordinate_pixel = camera1.world_to_pixel(coordinate_world)
# coordinate_pixel /= coordinate_pixel[2]
#
# print(coordinate_pixel)
#
# coordinate_world_back = camera1.pixel_to_world(coordinate_pixel)
# # coordinate_world_back[3] = 0
#
# coordinate_world_with_depth = camera1.pixel_depth_to_world(coordinate_pixel, 100)
#
# print('coordinate_world_with_depth')
# print(coordinate_world_with_depth)
#
# print(coordinate_world_back)
#
# coordinate_pixel_again = camera1.world_to_pixel(coordinate_world_back)
#
# print(coordinate_pixel_again)
#
# print(camera1.generate_camera_position_orientation_from_R_T()[1])

camera_test()
