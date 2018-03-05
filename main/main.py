import numpy as np
import os

from frame import frame as fm
from world_model import world as wd
from camera import camera as cm
from detector import object_detector as od

cwd = os.path.dirname(os.path.realpath(__file__)) 

CKPT_FILE_PATH = os.path.abspath(cwd + '/../data/model_files/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb')
LABEL_FILE_PATH = os.path.abspath(cwd + '/../data/label_files/mscoco_label_map.pbtxt')
NUM_CLASS = 90

detector = None
K = None
camera = None


def setup():
    global K
    global camera

    K = np.asmatrix([[1.12265949e+03, 0.00000000e+00, 3.07322388e+02, 0.00000000e+00],
                     [0.00000000e+00, 1.11948374e+03, 5.50850887e+02, 0.00000000e+00],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T = [[1], [1], [1]]
    camera = cm.Camera(86500, 86500, 0.035, 3264 / 2, 2448 / 2, R, T)
    camera.K = K
    camera.update_M_M_pinv_by_K_RT()

    global detector
    detector = od.ObjectorDetector(CKPT_FILE_PATH, LABEL_FILE_PATH, NUM_CLASS)


def main(imagesL, imagesR):
    # set up the initial cameras 
    global camera
    global detector
    frame_list = fm.generate_raw_frame_chain_from_images(imagesL, imagesR, raw_camera=camera)
    fm.setup_first_frame_in_frame_chain(frame_list)
    fm.generate_set_kp_des_in_frame_chain(frame_list)
    fm.generate_set_depth_info_in_frame_chain(frame_list)
    fm.generate_set_motion_info_in_frame_chain(frame_list)
    fm.generate_set_camera_extrinsic_parameters_in_frame_chain(frame_list)
    fm.generate_set_detection_info_in_frame_chain(frame_list, detector)
    fm.generate_set_projections_in_frame_chain(frame_list)

    world = wd.World()
    world.add_projections(fm.collect_projections_from_frames(frame_list))
    world.unify_objects_projection_get_object()

    return world.get_json()


def test():
    from video_process import video_process
    VIDEO_PATH_L = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/video_files/test_left.mp4'
    VIDEO_PATH_R = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/video_files/test_right.mp4'
    VIDEO_PATH_C = '/Users/zijunyan/Downloads/video_cal.mp4'
    FRAME_DIR_PATH = '../../data/image_files'
    STEP = 30

    # images_cal = video_process.capture_frames_from_video(VIDEO_PATH_C, 5)

    imagesL = video_process.capture_frames_from_video(VIDEO_PATH_L, STEP)
    imagesR = video_process.capture_frames_from_video(VIDEO_PATH_R, STEP)

    setup()
    imagesLR = []
    for i in range(min(len(imagesL), len(imagesR))):
        imagesLR.append((imagesL[i], imagesR[i]))

    json = main(imagesLR)

    print(json)

    # fm.motion_check_plot(frames)
    # fm.object_depth_detection_check_plot()


# test()

