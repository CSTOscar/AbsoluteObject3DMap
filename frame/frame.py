import numpy as np
import copy
from detector.motion_detector import detect_motion
from detector.depth_detector import detect_depth
from detector.keypt_des_detector import detect_keypt_des


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
        self.kp_des_set = False

        # id for different frame checking in later world model
        self.id = Frame.get_unique_id()

        # the basic frame information: image, shape, next and previous frame, intrinsic camera
        self.imageL = imageL
        self.imageR = imageR
        if imageL.shape != imageR.shape:
            print('FATAL: image LR shapes are different ')
        self.shape = self.imageL.shape
        self.camera = raw_camera
        self.next_frame = None
        self.prev_frame = None

        # for depth and motion detection

        self.kp_left = None
        self.des_left = None
        self.kp_right = None
        self.des_right = None

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

    def generate_set_kp_des(self):
        if not self.kp_des_set:
            (self.kp_left, self.des_left), (self.kp_right, self.des_right) = detect_keypt_des(self)
            self.kp_des_set = True
        else:
            print('Warning: self.kp_des_set more than once')

    def generate_set_detection_info(self, detector):
        if not self.detection_info_generated:
            self.detection_info_generated = True
            self.detection_info = detector.detect_object(self.imageL)
        else:
            print('Warning: generate_set detection_info more than once')

    def generate_set_depth_info(self):
        if not self.depth_info_generated:
            if self.kp_des_set:
                self.depth_info = detect_depth(self)
                self.depth_info_generated = True
            else:
                print('FATAL: kp_des not set')
        else:
            print('Warning: generate_set depth_info more than once')

    # TODO: fit this to the slam implementation
    def generate_set_motion_info(self):
        if self.prev_frame_set and self.prev_frame.kp_des_set:
            if not self.motion_info_generated:
                self.motion_info = detect_motion(self)
                self.motion_info_generated = True
            else:
                print('Warning: generate_set depth_info more than once')
        else:
            print('FATAL: prev_frame is not set or prev_kp_des is not set')

    # TODO: fit this to the slam implementation
    def generate_update_camera_extrinsic_parameters_based_on_prev_frame(self):
        if self.prev_frame_set and self.motion_info_generated:
            if not self.camera_extrinsic_set:
                if self.prev_frame.camera_extrinsic_set:
                    self.camera.update_RT(self.prev_frame.camera.RT)
                    vec6 = detect_motion(self)
                    transition_vec = vec6[0:3]
                    rotation_vec = vec6[3:6]
                    self.camera.update_extrinsic_parameters_by_world_camera_transformation(transition_vec, rotation_vec)
                else:
                    print('FATAL: prev_frame camera extrinsic para is not set')
            else:
                print('Warning: camera_extrinsic_parameters set more than once')
        else:
            print('FATAL: prev_frame is not set (the first frame should also be set to None) '
                  'or the motion_info is not generated')

    def get_depths_in_pixel_box(self, pixel_box):
        if not self.depth_info_generated:
            print('FATAL: depth_info is not generated')
        else:
            depths = []
            for (x, y, d) in self.depth_info:
                x_low = pixel_box[0]
                x_high = pixel_box[2]
                y_low = pixel_box[1]
                y_high = pixel_box[3]
                if x_low <= x <= x_high and y_low <= y <= y_high:
                    depths.append(d)

            if len(depths) == 0:
                print('FATAL: no depth is in the detection box')

            return depths

    def generate_set_projections(self):
        if self.detection_info_generated and self.depth_info_generated and self.camera_extrinsic_set:
            x_num_pix = self.shape[0]
            y_num_pix = self.shape[1]

            detection_info = self.detection_info

            projections = []

            for detection in detection_info:
                box = detection['box']
                pixel_box = [int(box[i] * (x_num_pix if i % 2 == 0 else y_num_pix)) for i in range(4)]

                center = [[(pixel_box[0] + pixel_box[2]) // 2], [(pixel_box[1] + pixel_box[3]) // 2]]
                # TODO: depth_mean is problematic, find a better metric later!
                depths = self.get_depths_in_pixel_box(pixel_box)
                if len(depths) == 0:
                    # if no depth is detected in the box
                    continue
                else:
                    depth = np.mean()
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
