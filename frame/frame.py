import numpy as np
import world_model.object as world_object


class Frame:
    id_seed = 0

    @classmethod
    def get_unique_id(cls):
        cls.id_seed += 1
        return cls.id_seed

    def generate_frame_info_from_detection_slam_info(self):
        slam_info = self.slam_info
        detection_info = self.detection_info

        x_num_pix = self.x_num_pixel
        y_num_pix = self.y_num_pixel

        depth_map = slam_info['depth']

        detection_slam_info = []

        for detection in detection_info:
            detection_slam = {}

            box = detection['box']
            box_x0 = box[0]
            box_y0 = box[1]
            box_x1 = box[2]
            box_y1 = box[3]
            center = np.asmatrix(
                [[int((box_x0 + box_x1) * x_num_pix // 2)], [int((box_y0 + box_y1) * y_num_pix // 2)], [1.0]])
            x_len = int((box_x1 - box_x0) * x_num_pix)
            y_len = int((box_y1 - box_y0) * y_num_pix)
            pixel_box = [(box[i] * (x_num_pix if i % 2 == 0 else y_num_pix)) for i in range(4)]
            print(pixel_box)
            # TODO: this is problematic, find a better metric later!
            # print('x_0', int(box[0] * x_num_pix))
            # print('x_1', int(box[2] * x_num_pix))
            # print('y_0', int(box[1] * y_num_pix))
            # print('y_1', int(box[3] * y_num_pix))
            depth_map_slice = depth_map[int(box_x0 * x_num_pix):int(box_x1 * x_num_pix),
                              int(box_y0 * y_num_pix):int(box_y1 * y_num_pix)]
            depth_mean = np.mean(depth_map_slice)
            # print(depth_map_slice, 'empty slice' if depth_map_slice.size == 0 else '')

            detection_slam['depth'] = depth_mean
            detection_slam['center'] = center
            detection_slam['x_len'] = x_len
            detection_slam['y_len'] = y_len
            detection_slam['score'] = detection['score']
            detection_slam['class'] = detection['class']
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
        self.orientation = slam_info['orientation']
        self.camera = camera
        self.camera.update_extrinsic_parameters_by_camera_position_orientation(self.position, self.orientation)

        # properties for object generation
        self.detection_info = detection_info
        self.slam_info = slam_info

        self.detection_slam_info = self.generate_frame_info_from_detection_slam_info()

    def get_objects_with_confidence_more_than(self, confidence):
        detection_slam_info_filted = filter(lambda e: e['score'] > confidence, self.detection_slam_info)

        objects = list(map(
            lambda e: world_object.Object(self.id, e['class'], e['score'],
                                          self.camera.pixel_depth_to_world(e['center'], e['depth']),
                                          self.camera.get_cov_by_depth(e['depth'])), detection_slam_info_filted))
        return objects
