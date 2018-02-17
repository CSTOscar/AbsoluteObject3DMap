import numpy as np


def camera_rotation_translation_to_R_and_T(translation, rotation):
    R = np.linalg.inv(rotation)
    T = -R @ translation
    return R, T


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
        self.RT = self.generate_RT()

        # Matrix for perspective projection
        self.K = self.generate_K()

        # Matrix and its inverse for transformation + perspective projection
        self.M = self.K @ self.RT
        self.M_pinv = np.linalg.pinv(self.M)

    def generate_RT(self):
        RconT = np.concatenate((self.R, self.T), axis=1)
        return np.concatenate((RconT, np.asmatrix([0, 0, 0, 1])), axis=0)

    def generate_K(self):
        return np.asmatrix([[self.f * self.mx, 0, self.u, 0], [0, self.f * self.my, self.v, 0], [0, 0, 1, 0]])

    def generate_camera_position_orientation_from_R_T(self):
        R_inv = np.linalg.inv(self.R)
        return -R_inv @ self.T, R_inv

    def update_extrinsic_parameters_by_camera_position_orientation(self, position, orientation):
        self.R, self.T = camera_rotation_translation_to_R_and_T(position, orientation)

    def world_to_pixel(self, world_coordinate):
        ans = self.M @ world_coordinate
        ans /= ans[2]
        return self.M @ world_coordinate

    def pixel_to_world(self, pixel_coordinate):
        ans = self.M_pinv @ pixel_coordinate
        return ans

