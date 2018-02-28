import numpy as np
import functools
import cv2


# WARNING: this is a right handed coordinate system
#
def position_direction_rotation_output_adapter(method):
    @functools.wraps(method)
    def adapted_method(*args, **kwargs):
        results = method(*args, **kwargs)
        results = results
        for i in range(len(results)):
            result = results[i]
            if isinstance(result, tuple) or isinstance(result, list) or \
                    isinstance(result, np.matrix) or isinstance(result, np.ndarray):
                result = np.asarray(result).flatten()
                # print(arg)
                result_len = len(result)
                if result_len == 3:
                    result = result
                elif result_len == 9:
                    result = result.reshape((3, 3)).tolist()
                else:
                    print('FATAL: result_len != 3 or 9 in', method.__name__)
                yield result

    adapted_method.original = method
    return adapted_method


def position_direction_rotation_input_adapter(method):
    @functools.wraps(method)
    def adapted_method(*args, **kwargs):
        # print('origin: ', args)
        args = list(args)
        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, tuple) or isinstance(arg, list) or \
                    isinstance(arg, np.matrix) or isinstance(arg, np.ndarray):
                arg = np.asarray(arg).flatten()
                # print(arg)
                arg_len = len(arg)
                if arg_len == 3:
                    arg = np.asmatrix(arg.reshape((3, 1)))
                elif arg_len == 9:
                    arg = np.asmatrix(arg.reshape((3, 3)))
            # else:
            #     print('FATAL: invalid input for ', method.__name__)
            args[i] = arg

        # print('adapted: ', args)

        return method(*args, **kwargs)

    adapted_method.original = method

    return adapted_method


def coordinates_input_output_adapter(method):
    @functools.wraps(method)
    def adapted_method(*args, **kwargs):
        args = list(args)
        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, tuple) or isinstance(arg, list) or \
                    isinstance(arg, np.matrix) or isinstance(arg, np.ndarray):
                arg = np.asarray(arg).flatten()
                # print(arg)
                arg = np.append(arg, [1.0])
                arg = np.asmatrix(arg.reshape((len(arg), 1)))
            # else:
            #     print('FATAL: invalid input for ', method.__name__)
            args[i] = arg

        # print(args)

        def return_adapted(*args, **kwargs):
            # TODO: this just handles single output, expand later
            result = method(*args, **kwargs)
            if isinstance(result, np.ndarray) or isinstance(result, np.matrix):
                # print(result)
                result = np.asarray(result).flatten()
                # TODO: this might lead to divide by zero, think about it
                result = result / result[-1]
                result = result.tolist()[:-1]
            else:
                print('FATAL: invalid output for ', method.__name__, 'check camera.py, programmers fault')
            return result

        return return_adapted(*args, **kwargs)

    adapted_method.original = method

    return adapted_method


class Camera:
    DEFAULT_CORNER_SUB_PIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def calibrate_by_images_and_grid_length(self, image, length, criteria=DEFAULT_CORNER_SUB_PIX_CRITERIA):

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(image_gray, (7, 6), None)

        if ret:
            image_points = cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1), criteria)
            image_points = [image_points]
            object_points = np.zeros((6 * 7, 3), np.float32)
            object_points[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
            object_points *= length
            object_points = [object_points]

            ret, intrinsic_matrix, distortion, rotation_vectors, transformation_vectors = cv2.calibrateCamera(
                object_points, image_points, image_gray.shape[::-1],
                None, None)

            print(image_points)
            print(object_points)
            print(intrinsic_matrix)
            print(rotation_vectors)
            print(transformation_vectors)

            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vectors[0])
            rotation_matrix = np.asmatrix(rotation_matrix)
            transformation_vector = transformation_vectors[0]
            transformation_vector = np.asmatrix(transformation_vector.reshape((3, 1)))
            intrinsic_matrix = np.append(intrinsic_matrix, [[0], [0], [0]], axis=1)

            if ret:
                self.K = intrinsic_matrix
                self.R = rotation_matrix
                self.T = transformation_vector
                print(self.K)
                print(self.R)
                print(self.T)
                self.RT = Camera.generate_RT_from_R_T(self.R, self.T)
                self.M = self.K @ self.RT
                self.M_pinv = np.linalg.pinv(self.M)

    @staticmethod
    def camera_position_rotation_to_R_T(position, rotation):
        # print('camera_position_rotation_to_R_T ', position, rotation)
        R = np.linalg.inv(rotation)
        T = -R @ position
        return R, T

    @staticmethod
    def camera_position_direction_to_R_T(position, direction):
        rotation = Camera.generate_rotation_from_direction(direction)
        return Camera.camera_position_rotation_to_R_T(position, rotation)

    @staticmethod
    def generate_RT_from_R_T(R, T):
        RconT = np.concatenate((R, T), axis=1)
        return np.concatenate((RconT, np.asmatrix([0, 0, 0, 1])), axis=0)

    @staticmethod
    def generate_K_from_mx_my_f_u_v(mx, my, f, u, v):
        return np.asmatrix([[f * mx, 0, u, 0],
                            [0, f * my, v, 0],
                            [0, 0, 1, 0]])

    @staticmethod
    def generate_rotation_from_direction(D):
        D = D / np.linalg.norm(D)
        Z = np.asmatrix([[0.0], [0.0], [1.0]])
        C = Z.T @ D
        V = np.cross(Z.flatten(), D.flatten()).reshape((3, 1))
        Vx = np.asmatrix([[0.0, -V[2, 0], V[1, 0]],
                          [V[2, 0], 0.0, -V[0, 0]],
                          [-V[1, 0], V[0, 0], 0.0]])
        I = np.asmatrix([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])
        # S = np.linalg.norm(V)
        if C != -1.0:
            R_inv = I + Vx + Vx @ Vx / (1 + C)
        else:
            R_inv = -I
        return R_inv

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
        self.RT = Camera.generate_RT_from_R_T(self.R, self.T)

        # Matrix for perspective projection
        self.K = Camera.generate_K_from_mx_my_f_u_v(self.mx, self.my, self.f, self.u, self.v)

        # Matrix and its inverse for transformation + perspective projection
        self.M = self.K @ self.RT
        self.M_pinv = np.linalg.pinv(self.M)

        self.cov = np.asmatrix([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])

    @position_direction_rotation_output_adapter
    def generate_camera_position_rotation_from_R_T(self):
        R_inv = np.linalg.inv(self.R)
        return -R_inv @ self.T, R_inv

    @position_direction_rotation_output_adapter
    def generate_camera_position_direction_from_R_T(self):
        position, rotation = self.generate_camera_position_rotation_from_R_T()
        direction = rotation @ np.asmatrix([[0.0], [0.0], [1.0]])
        return position, direction

    @position_direction_rotation_input_adapter
    def update_extrinsic_parameters_by_camera_position_rotation(self, position, rotation):
        # print('update_extrinsic_parameters_by_camera_position_rotation', position, rotation)
        self.R, self.T = Camera.camera_position_rotation_to_R_T(position, rotation)
        self.RT = Camera.generate_RT_from_R_T(self.R, self.T)
        self.M = self.K @ self.RT
        self.M_pinv = np.linalg.pinv(self.M)

    @position_direction_rotation_input_adapter
    def update_extrinsic_parameters_by_camera_position_direction(self, position, direction):
        rotation = Camera.generate_rotation_from_direction(direction)
        self.update_extrinsic_parameters_by_camera_position_rotation(position, rotation)

    @coordinates_input_output_adapter
    def world_to_pixel(self, world_coordinate):
        ans = self.M @ world_coordinate
        ans /= ans[2]
        return self.M @ world_coordinate

    @coordinates_input_output_adapter
    def pixel_to_world(self, pixel_coordinate):
        ans = self.M_pinv @ pixel_coordinate
        if ans[-1] == 0:
            # TODO: argue this is mathematically true, write the intuition in the report
            ans[-1] = 1.0
        return ans

    @coordinates_input_output_adapter
    def pixel_depth_to_world(self, pixel_coordinate, depth):
        P, O = self.generate_camera_position_rotation_from_R_T.original(self)
        Z = O @ np.asmatrix([[0], [0], [1]])

        # E3 = self.pixel_to_world(pixel_coordinate[:-1] / pixel_coordinate[-1])
        # E3 = np.asarray(E3).reshape((3, 1))
        # E3 = np.asmatrix(E3)

        E4 = self.pixel_to_world.original(self, pixel_coordinate)
        if E4[-1] == 0.0:
            # TODO: argue that if the E_4 is 0, the camera position must be at 0
            # this is handled in pixel_to_world
            E3 = E4[:-1]
        else:
            E3 = E4[:-1] / E4[-1]

        D0 = (E3 - P)
        D0 = D0 / np.linalg.norm(D0)
        D = D0 * (-1 if (D0.T @ Z)[0][0] < 0 else 1)

        world_coordinate = P + depth * D
        world_coordinate = np.append(world_coordinate, [[1.0]], axis=0)

        return world_coordinate

    @DeprecationWarning
    def get_cov_by_depth(self, depth):
        return depth * self.cov
