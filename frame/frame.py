import numpy as np
import world_model.object as world_object


class Frame:
    id_seed = 0

    @classmethod
    def get_unique_id(cls):
        cls.id_seed += 1
        return cls.id_seed

    def add_pixel_W_H_center_depth_to_detection_info(self, depth_map):
        x_num_pix = self.x_num_pixel
        y_num_pix = self.y_num_pixel
        for record in self.detection_info:
            box = record['box']
            center = (int((box[0] + box[2]) * x_num_pix // 2), int((box[1] + box[3]) * y_num_pix // 2))
            xy_len = (int((box[2] - box[0]) * x_num_pix), int((box[3] - box[1]) * y_num_pix))
            # TODO: this is problematic, find a better metric later!
            record['depth'] = np.mean(depth_map[int(box[0] * x_num_pix):int(box[2] * x_num_pix)][
                                      int(box[1] * x_num_pix):int(box[3] * x_num_pix)])
            record['center'] = center
            record['xy_len'] = xy_len
        return record

    def __init__(self, x_num_pixel, y_num_pixel, position, orientation, detection_info, depth_map, camera):
        self.id = Frame.get_unique_id()
        self.x_num_pixel = x_num_pixel
        self.y_num_pixel = y_num_pixel
        self.position = position
        self.orientation = orientation
        self.detection_info = detection_info
        self.detection_info = self.add_pixel_W_H_center_depth_to_detection_info(depth_map)
        self.camera = camera.update_extrinsic_parameters_by_camera_position_orientation(position, orientation)

    def get_objects_with_confidence_more_than(self, confidence):
        world_object.Object()
        objects = list(map(
            lambda e: world_object.Object(self.id, e['class'], e['score'],
                                          self.camera.pixel_depth_to_world(e['center'], e['depth']),
                                          self.camera.get_cov_by_depth(e['depth'])),
            filter(lambda e: e['score'] > confidence, self.detection_info)))
        return objects
