import numpy as np
import world_model.object_projection as world_object
from camera import camera as camera_


class Frame:
    id_seed = 0

    @classmethod
    def get_unique_id(cls):
        cls.id_seed += 1
        return cls.id_seed

    def generate_detection_slam_info(self):
        slam_info = self.slam_info
        detection_info = self.detection_info

        x_num_pix = self.x_num_pixel
        y_num_pix = self.y_num_pixel

        depth_map = slam_info['depth']

        detection_slam_info = []

        for detection in detection_info:
            detection_slam = {}

            box = detection['box']
            pixel_box = [int(box[i] * (x_num_pix if i % 2 == 0 else y_num_pix)) for i in range(4)]

            center = np.asmatrix(
                [[(pixel_box[0] + pixel_box[2]) // 2], [(pixel_box[1] + pixel_box[3]) // 2]])
            # x_len = pixel_box[2] - pixel_box[0]
            # y_len = pixel_box[3] - pixel_box[1]

            # print(pixel_box)
            # TODO: this is problematic, find a better metric later!
            # print('x_0', int(box[0] * x_num_pix))
            # print('x_1', int(box[2] * x_num_pix))
            # print('y_0', int(box[1] * y_num_pix))
            # print('y_1', int(box[3] * y_num_pix))
            depth_map_slice = depth_map[pixel_box[0]:pixel_box[2], pixel_box[1]:pixel_box[3]]
            # print(depth_map.shape)
            depth_mean = np.mean(depth_map_slice)
            # print(depth_map_slice, 'empty slice' if depth_map_slice.size == 0 else '')

            detection_slam['depth'] = depth_mean
            detection_slam['center'] = center
            # detection_slam['x_len'] = x_len
            # detection_slam['y_len'] = y_len
            detection_slam['score'] = detection['score']
            detection_slam['class'] = int(detection['class'])
            detection_slam['pixel_box'] = pixel_box
            detection_slam_info.append(detection_slam)

        return detection_slam_info

    def __init__(self, detection_info, slam_info, camera):
        depth_map = slam_info['depth']
        shape = depth_map.shape
        x_num_pixel = shape[0]
        y_num_pixel = shape[1]

        # property of the frame
        self.id = Frame.get_unique_id()
        self.x_num_pixel = x_num_pixel
        self.y_num_pixel = y_num_pixel
        self.position = slam_info['position']
        self.direction = slam_info['direction']
        self.camera = camera
        self.camera.update_extrinsic_parameters_by_camera_position_direction(self.position, self.direction)

        # properties for object generation
        self.detection_info = detection_info
        self.slam_info = slam_info

        self.detection_slam_info = self.generate_detection_slam_info()

    def get_objects_projection_with_confidence_more_than(self, confidence):
        detection_slam_info_filted = filter(lambda e: e['score'] > confidence, self.detection_slam_info)
        # print(list(detection_slam_info_filted)[0]['center'])
        objects = list(map(
            lambda e: Frame.new_object_projection(self.id, e['class'], e['score'],
                                                  self.camera.pixel_depth_to_world(e['center'], e['depth']),
                                                  self.camera.get_cov_by_depth(e['depth'])),
            detection_slam_info_filted))
        return objects

    @staticmethod
    def new_object_projection(frame_id, clazz, score, world_coordinate, err_cov):
        return {'frame_id': frame_id, 'class': clazz, 'score': score, 'position': world_coordinate, 'error': err_cov}
