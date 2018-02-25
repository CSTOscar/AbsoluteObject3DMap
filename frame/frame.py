import numpy as np
import copy
import world_model.object_projection as world_object
from camera import camera as camera_


# TODO: debug, test and check for error message and logic

def new_object_projection(frame_id, clazz, score, world_coordinate, scale):
    return {'frame_id': frame_id, 'class': clazz, 'score': score,
            'position': world_coordinate, 'scale': scale}


def generate_raw_frame_chain_from_images(imageL_list, imageR_list, raw_camera):
    if len(imageL_list) != len(imageR_list):
        print("Warning: len(imageL_list) != len(imageR_list)")

    frame_list = []
    for i in range(min(len(imageL_list), len(imageR_list))):
        raw_camera_copy = copy.deepcopy(raw_camera)
        frame = Frame(imageL_list[i], imageR_list[i], raw_camera_copy)
        frame_list.append(frame)

    frame_list[0].set_prev_frame(None)
    frame_list[0].set_next_frame(frame_list[1])
    for i in range(1, len(frame_list) - 1):
        frame_list[i].set_prev_frame(frame_list[i - 1])
        frame_list[i].set_next_frame(frame_list[i + 1])
    frame_list[-1].set_prev_frame(frame_list[len(frame_list) - 1])
    frame_list[-1].set_next_frame(None)

    return frame_list


class Frame:
    id_seed = 0

    @classmethod
    def get_unique_id(cls):
        cls.id_seed += 1
        return cls.id_seed

    def __init__(self, imageL, imageR, raw_camera):
        # booleans for progress check
        # ordered roughly in dependence
        self.prev_frame_set = False
        self.next_frame_set = False
        self.detection_info_generated = False
        self.depth_info_generated = False
        self.motion_info_generated = False
        self.camera_extrinsic_set = False

        # id for different frame checking in later world model
        self.id = Frame.get_unique_id()

        # the basic frame information: image, shape, next and previous frame, intrinsic camera
        self.imageL = imageL
        self.imageR = imageR
        self.shape = self.imageL.shape
        self.camera = raw_camera
        self.next_frame = None
        self.prev_frame = None

        # property of the frame: detection, depth
        self.detection_info = None
        self.depth_info = None

        # property of the frame and the previous frame: motion of camera
        self.motion_info = None

        # sef of object projections which will be passed to world reconstruction
        self.projections = None

    def set_prev_frame(self, frame):
        if not self.prev_frame_set:
            self.prev_frame_set = True
            self.prev_frame = frame
        else:
            print('Warning: set prev_frame more than once')

    def set_next_frame(self, frame):
        if not self.next_frame_set:
            self.next_frame_set = True
            self.next_frame = frame
        else:
            print('Warning: set prev_frame more than once')

    def generate_set_detection_info(self, detector):
        if not self.detection_info_generated:
            self.detection_info_generated = True
            self.detection_info = detector.detect_object(self.imageL)
        else:
            print('Warning: generate_set detection_info more than once')

    def generate_set_depth_info(self, detector):
        if not self.depth_info_generated:
            depth_info = detector.detect_depth(self.imageL, self.imageR)
            if depth_info.shape == self.shape:
                self.depth_info_generated = True
                self.depth_info = depth_info
            else:
                print('FATAL: depth_info shape does not match the frame shape')
        else:
            print('Warning: generate_set depth_info more than once')

    def generate_set_motion_info(self, detector):
        if self.prev_frame_set:
            if not self.motion_info_generated:
                motion_info = detector.detect_motion(self.imageL, self.prev_frame.ImageL)
                # TODO: sanity check
                self.motion_info = motion_info
            else:
                print('Warning: generate_set depth_info more than once')
        else:
            print('FATAL: prev_frame is not set (the first frame should also be set to None)')

    def generate_update_camera_extrinsic_parameters_based_on_prev_frame(self):
        if self.prev_frame_set and self.motion_info_generated:
            if not self.camera_extrinsic_set:
                if self.prev_frame.camera_extrinsic_set:
                    prev_camera = self.prev_frame.camera
                    prev_camera_position, prev_camera_direction = \
                        prev_camera.generate_camera_position_direction_from_R_T()
                    transition = self.motion_info['transition']
                    rotation = self.motion_info['rotation']
                    curr_camera_position = transition @ prev_camera_position
                    curr_camera_direction = rotation @ prev_camera_direction
                    self.camera_extrinsic_set = True
                    self.camera.update_extrinsic_parameters_by_camera_position_direction(curr_camera_position,
                                                                                         curr_camera_direction)
                else:
                    print('FATAL: prev_frame camera extrinsic para is not set')
            else:
                print('Warning: camera_extrinsic_parameters set more than once')
        else:
            print('FATAL: prev_frame is not set (the first frame should also be set to None) '
                  'or the motion_info is not generated')

    def generate_set_projections(self):
        if self.detection_info_generated and self.depth_info_generated and self.camera_extrinsic_set:
            x_num_pix = self.shape[0]
            y_num_pix = self.shape[1]

            depth_map = self.depth_info
            detection_info = self.detection_info

            projections = []

            for detection in detection_info:
                box = detection['box']
                pixel_box = [int(box[i] * (x_num_pix if i % 2 == 0 else y_num_pix)) for i in range(4)]

                center = np.asmatrix([[(pixel_box[0] + pixel_box[2]) // 2], [(pixel_box[1] + pixel_box[3]) // 2]])
                # TODO: depth_mean is problematic, find a better metric later!
                depth_map_slice = depth_map[pixel_box[0]:pixel_box[2], pixel_box[1]:pixel_box[3]]
                depth = np.mean(depth_map_slice)
                position = self.camera.pixel_depth_to_world(center, depth)

                # TODO: implement scale later
                scale = max(pixel_box)
                projections.append(
                    new_object_projection(self.id, detection['class'], detection['score'], position, scale))
        else:
            print(
                'FATAL: self.detection_info_generated and self.depth_info_generated and self.camera_extrinsic_set is false')

    @DeprecationWarning
    def get_objects_projection_with_confidence_more_than(self, confidence):
        detection_slam_info_filted = filter(lambda e: e['score'] > confidence, self.detection_depth_motion_info)
        objects = list(map(
            lambda e: Frame.new_object_projection(self.id, e['class'], e['score'],
                                                  self.camera.pixel_depth_to_world(e['center'], e['depth']),
                                                  self.camera.get_cov_by_depth(e['depth'])),
            detection_slam_info_filted))
        return objects
