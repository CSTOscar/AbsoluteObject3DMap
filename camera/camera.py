import numpy as np


def camera_translation_rotation_to_R_and_T(translation, rotation):
    R = np.linalg.inv(rotation)
    T = -R @ translation
    return R, T


def generate_RT_from_R_T(R, T):
    RconT = np.concatenate((R, T), axis=1)
    return np.concatenate((RconT, np.asmatrix([0, 0, 0, 1])), axis=0)


def generate_K_from_mx_my_f_u_v(mx, my, f, u, v):
    return np.asmatrix([[f * mx, 0, u, 0], [0, f * my, v, 0], [0, 0, 1, 0]])


class Camera:
    def __init__(self, mx, my, f, u, v, R, T):
        # Intrinsic parameters
        self.mx = mx
        self.my = my
        self.f = f
        self.u = u
        self.v = v

        # Extrinsic parameters
        # R^-1 is the rotation matrix for how camera is rotated
        # T is the world's origin coordinates in camera's coordinate system
        self.R = R
        self.T = T

        # Matrix for coordinate transformation
        self.RT = generate_RT_from_R_T(self.R, self.T)

        # Matrix for perspective projection
        self.K = generate_K_from_mx_my_f_u_v(self.mx, self.my, self.f, self.u, self.v)

        # Matrix and its inverse for transformation + perspective projection
        self.M = self.K @ self.RT
        self.M_pinv = np.linalg.pinv(self.M)

        self.cov = np.asmatrix([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])

    def generate_camera_position_orientation_from_R_T(self):
        R_inv = np.linalg.inv(self.R)
        return -R_inv @ self.T, R_inv

    def update_extrinsic_parameters_by_camera_position_orientation(self, position, orientation):
        self.R, self.T = camera_translation_rotation_to_R_and_T(position, orientation)
        self.RT = generate_RT_from_R_T(self.R, self.T)
        self.M = self.K @ self.RT
        self.M_pinv = np.linalg.pinv(self.M)

    def world_to_pixel(self, world_coordinate):
        ans = self.M @ world_coordinate
        ans /= ans[2]
        return self.M @ world_coordinate

    def pixel_to_world(self, pixel_coordinate):
        ans = self.M_pinv @ pixel_coordinate
        return ans

    def pixel_depth_to_world(self, pixel_coordinate, depth):
        print('pixel coordinate: ', pixel_coordinate)
        print('depth', depth)
        P, O = self.generate_camera_position_orientation_from_R_T()
        print('camera position:', P)
        Z = O @ np.asmatrix([[0], [0], [1]])
        E4 = self.pixel_to_world(pixel_coordinate)
        # TODO: argue that if the E_4 is 0, the camera position must be at 0
        E3 = E4[0:3] / (E4[3][0] if E4[3][0] != 0 else 1.0)
        print('E3', E3)
        print('E4', E4)
        D0 = (E3 - P)
        D0 = D0 / np.linalg.norm(D0)
        D = D0 * (-1 if (D0.T @ Z)[0][0] < 0 else 1)
        print('D: ', D)
        world_coordinate = P + depth * D
        print('world coordinate: ', world_coordinate)
        return world_coordinate

    def get_cov_by_depth(self, depth):
        return depth * self.cov
