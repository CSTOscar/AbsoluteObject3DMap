import copy
from detector.motion_detector import detect_motion
from detector.depth_detector import detect_depth
from detector.keypt_des_detector import detect_keypt_des
from detector.motion_detector import MotionDetectionFailed
from detector.depth_detector import DepthDetectionFailed
import matplotlib.pyplot as plt
import numpy as np
import cv2


# TODO: debug, test and check for error message and logic

def new_object_projection(frame_id, clazz, score, world_coordinate, scale, orientation):
    return {'frame_id': frame_id, 'class': clazz, 'score': score,
            'position': world_coordinate, 'size': scale, 'orientation': orientation}


def motion_check_plot(frames):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    XYZUVW = []
    for i, frame in enumerate(frames):
        position, direction = frame.camera.generate_camera_position_direction_from_R_T()
        position.extend(direction)
        XYZUVW.append(position)

    XYZUVW_T = np.array(XYZUVW).T
    ax.quiver(XYZUVW_T[0], XYZUVW_T[1], XYZUVW_T[2], XYZUVW_T[3], XYZUVW_T[4], XYZUVW_T[5])
    plt.show()


def object_depth_detection_check_plot(frames):
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    images = []
    for frame in frames:
        image = copy.copy(frame.imageL)
        if frame.depth_info is not None:
            for (xy, depth) in frame.depth_info:
                # print(image.shape, xy)
                cv2.circle(image, tuple(xy), 2, RED, thickness=4)
        for obj_detect in frame.detection_info:
            if obj_detect['score'] >= 0.7:
                if 'pixel_box' in obj_detect.keys():
                    box = obj_detect['pixel_box']
                    cv2.rectangle(image, tuple(box[0:2]), tuple(box[2:4]), BLUE, thickness=4)
        images.append(image)

    return images


def collect_projections_from_frames(frames, confidence=0.8):
    print('collect_projections_from_frames starts')
    projections = []
    for frame in frames:
        projections.extend(frame.get_objects_projection_with_confidence_more_than(confidence))
        print('progress check: ', frame.id, ' done.')
    return projections


def generate_raw_frame_chain_from_images(imageL_list, imageR_list, raw_camera):
    if len(imageL_list) != len(imageR_list):
        print("Warning: len(imageL_list) != len(imageR_list)")

    frame_list = []
    for i in range(min(len(imageL_list), len(imageR_list))):
        raw_camera_copy = copy.deepcopy(raw_camera)
        frame = Frame(imageL_list[i], imageR_list[i], raw_camera_copy)
        frame_list.append(frame)

    frame_list[0].set_prev_frame(None)
    # if there is more set one frame
    if len(frame_list) > 1:
        frame_list[0].set_next_frame(frame_list[1])
        for i in range(1, len(frame_list) - 1):
            frame_list[i].set_prev_frame(frame_list[i - 1])
            frame_list[i].set_next_frame(frame_list[i + 1])
        frame_list[-1].set_prev_frame(frame_list[-2])
        frame_list[-1].set_next_frame(None)
    else:
        frame_list[0].set_next_frame(None)

    return frame_list


def generate_set_kp_des_in_frame_chain(frames):
    print('generate_set_kp_des_in_frame_chain starts')
    for frame in frames:
        frame.generate_set_kp_des()
        print('progress check: ', frame.id, ' done.')


def generate_set_detection_info_in_frame_chain(frames, detector):
    print('generate_set_detection_info_in_frame_chain starts')

    for frame in frames:
        frame.generate_set_detection_info(detector)
        print('progress check: ', frame.id, ' done.')


def generate_set_depth_info_in_frame_chain(frames):
    print('generate_set_depth_info_in_frame_chain starts')

    for frame in frames:
        frame.generate_set_depth_info()
        print('progress check: ', frame.id, ' done.')


def generate_set_motion_info_in_frame_chain(frames):
    print('generate_set_motion_info_in_frame_chain starts')

    if len(frames) == 1:
        print('A single frame does not need motion detection')
    for frame in frames[1:]:
        frame.generate_set_motion_info()
        print('progress check: ', frame.id, ' done.')


def generate_set_camera_extrinsic_parameters_in_frame_chain(frames):
    print('generate_set_camera_extrinsic_parameters_in_frame_chain starts')

    for frame in frames[1:]:
        frame.generate_update_camera_extrinsic_parameters_based_on_prev_frame()
        print('progress check: ', frame.id, ' done.')


def generate_set_projections_in_frame_chain(frames):
    print('generate_set_projections_in_frame_chain starts')
    for frame in frames:
        frame.generate_set_projections()
        print('progress check: ', frame.id, ' done.')


def setup_first_frame_in_frame_chain(frames):
    frame = frames[0]
    frame.camera_extrinsic_set = True
    frame.motion_info = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    frame.motion_info_generated = True


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
        self.projections_generated = False

        # id for different frame checking in later world model
        self.id = Frame.get_unique_id()

        # the basic frame information: image, shape, next and previous frame, intrinsic camera
        self.imageL = imageL
        self.imageR = imageR
        if imageL.shape != imageR.shape:
            print('FATAL in frame', self.id, ' : image LR shapes are different')
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
        # box[yx,yx]
        self.detection_info = None

        # [([yx],d)]
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
            print('Warning in frame', self.id, ' : set prev_frame more than once')

    def set_next_frame(self, frame):
        if not self.next_frame_set:
            self.next_frame_set = True
            self.next_frame = frame
        else:
            print('Warning in frame', self.id, ' : set prev_frame more than once')

    def generate_set_kp_des(self):
        if not self.kp_des_set:
            (self.kp_left, self.des_left), (self.kp_right, self.des_right) = detect_keypt_des(self)
            self.kp_des_set = True
        else:
            print('Warning in frame', self.id, ' : self.kp_des_set more than once')

    def generate_set_detection_info(self, detector):
        if not self.detection_info_generated:
            self.detection_info_generated = True
            self.detection_info = detector.detect_object(self.imageL)
        else:
            print('Warning in frame', self.id, ' : generate_set detection_info more than once')

    def generate_set_depth_info(self):
        if not self.depth_info_generated:
            if self.kp_des_set:
                try:
                    self.depth_info = detect_depth(self)
                    self.depth_info_generated = True
                except DepthDetectionFailed as depth_detection_failed:
                    print('FATAL: ', depth_detection_failed.args)
                    self.depth_info = []
                    self.depth_info_generated = True
            else:
                print('FATAL in frame', self.id, ' : kp_des_set ', self.kp_des_set)
        else:
            print('Warning in frame', self.id, ' : generate_set depth_info more than once')

    # TODO: fit this to the slam implementation
    def generate_set_motion_info(self):
        if self.prev_frame_set and self.prev_frame.kp_des_set:
            if not self.motion_info_generated:
                try:
                    self.motion_info = detect_motion(self)
                    self.motion_info_generated = True
                except MotionDetectionFailed as motion_detection_fail:
                    print('FATAL:  in frame', self.id, ' ', motion_detection_fail.args)
                    self.motion_info = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    self.motion_info_generated = True

            else:
                print('Warning in frame', self.id, ' : generate_set depth_info more than once')
        else:
            print('FATAL in frame', self.id, ' : prev_frame_set, prev_kp_des_generated', self.prev_frame_set,
                  self.prev_frame.kp_des_set)

    # TODO: fit this to the slam implementation

    def generate_update_camera_extrinsic_parameters_based_on_prev_frame(self):
        # def yx_to_xy(vector):
        #     temp = vector[0]
        #     vector[0] = vector[1]
        #     vector[1] = temp
        #     return vector

        if self.prev_frame_set and self.motion_info_generated:
            if not self.camera_extrinsic_set:
                if self.prev_frame.camera_extrinsic_set:
                    self.camera.update_RT(self.prev_frame.camera.RT)
                    vec6 = self.motion_info
                    transition_vec = (vec6[0:3])
                    rotation_vec = (vec6[3:6])
                    self.camera.update_extrinsic_parameters_by_world_camera_transformation(transition_vec,
                                                                                           rotation_vec)
                    self.camera_extrinsic_set = True
                else:
                    print('FATAL in frame', self.id, ' : prev_frame camera extrinsic para is not set')
            else:
                print('Warning in frame', self.id, ' : camera_extrinsic_parameters set more than once')
        else:
            print('FATAL in frame', self.id, ' : prev_frame_set', self.prev_frame_set)

    def get_depths_in_pixel_box(self, pixel_box):
        if not self.depth_info_generated:
            print('FATAL: depth_info is not generated')
            return []
        else:
            depths = []
            for (xy, d) in self.depth_info:
                x = xy[0]
                y = xy[1]
                x_low = pixel_box[0]
                x_high = pixel_box[2]
                y_low = pixel_box[1]
                y_high = pixel_box[3]
                if x_low <= x <= x_high and y_low <= y <= y_high:
                    depths.append(d)

            # if len(depths) == 0:
            #     print('Warning: no depth is in the detection box')

            return depths

    def find_size(self, pixel_box, depth):
        camera = self.camera
        pt1 = [pixel_box[0], pixel_box[1]]
        pt2 = [pixel_box[2], pixel_box[3]]
        pt3 = [pixel_box[2], pixel_box[1]]

        ptw1 = np.asarray(camera.pixel_depth_to_world(pt1, depth))
        ptw2 = np.asarray(camera.pixel_depth_to_world(pt2, depth))
        ptw3 = np.asarray(camera.pixel_depth_to_world(pt3, depth))

        scale = max([np.linalg.norm(ptw2 - ptw1), np.linalg.norm(ptw3 - ptw1)])
        return scale

    def generate_set_projections(self):
        camera_position, _ = self.camera.generate_camera_position_direction_from_R_T()
        if self.detection_info_generated and self.depth_info_generated and self.camera_extrinsic_set:
            if not self.projections_generated:
                y_num_pix = self.shape[0]
                x_num_pix = self.shape[1]

                detection_info = self.detection_info

                projections = []

                for detection in detection_info:
                    box = detection['box']
                    pixel_box = [int(box[i] * (x_num_pix if i % 2 == 0 else y_num_pix)) for i in range(4)]
                    detection['pixel_box'] = pixel_box
                    center = [[(pixel_box[0] + pixel_box[2]) // 2], [(pixel_box[1] + pixel_box[3]) // 2]]
                    # TODO: depth_mean is problematic, find a better metric later!
                    depths = self.get_depths_in_pixel_box(pixel_box)
                    if len(depths) == 0:
                        continue
                    else:
                        depth = np.mean(depths)
                        position = self.camera.pixel_depth_to_world(center, depth)

                    scale = self.find_size(pixel_box, depth)

                    orientation = np.array(position) - np.array(camera_position)
                    orientation = orientation / np.linalg.norm(orientation)

                    projections.append(
                        new_object_projection(self.id, detection['class'], detection['score'], position, scale,
                                              orientation))

                self.projections = projections
                self.projections_generated = True

            else:
                print('Warning in frame', self.id, ' : projections generated more than once')
        else:
            print(
                'FATAL in frame', self.id,
                ' : detection_info_generated, depth_info_generated,camera_extrinsic_set ',
                self.detection_info_generated, self.depth_info_generated, self.camera_extrinsic_set)

    def get_objects_projection_with_confidence_more_than(self, confidence):
        if self.projections_generated:
            return list(filter(lambda e: e['score'] >= confidence, self.projections))
        else:
            print('Warning in frame', self.id, ' : not projections_generated')
            return []
