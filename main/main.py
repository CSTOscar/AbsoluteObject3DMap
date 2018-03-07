import numpy as np
import os

from frame import frame as fm
from world_model import world as wd
from camera import camera as cm
from detector import object_detector as od
import cv2
from PIL import Image

cwd = os.path.dirname(os.path.realpath(__file__))

CKPT_FILE_PATH = os.path.abspath(
    cwd + '/../data/model_files/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb')
LABEL_FILE_PATH = os.path.abspath(cwd + '/../data/label_files/mscoco_label_map.pbtxt')
NUM_CLASS = 90

detector = None
K = None
camera = None

def setup(K0=None):
    global K
    global camera

    # K = np.asmatrix([[1.12265949e+03, 0.00000000e+00, 3.07322388e+02, 0.00000000e+00],
    #                  [0.00000000e+00, 1.11948374e+03, 5.50850887e+02, 0.00000000e+00],
    #                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])
    if K0 is not None:
        K = K0
    else:
        # K = np.asmatrix([[2.01078286e+03, 0.00000000e+00, 4.93720699e+02, 0.00000000e+00],
        #                  [0.00000000e+00, 2.02736414e+03, 1.01140977e+03, 0.00000000e+00],
        #                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])
        # new K value for 1080x1440
        K = np.asmatrix([[1.28953679e+03, 0.00000000e+00, 5.34768975e+02, 0.00000000e+00],
                         [0.00000000e+00, 1.29259578e+03, 7.08680476e+02, 0.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T = [[1], [1], [1]]
    camera = cm.Camera(86500, 86500, 0.035, 3264 / 2, 2448 / 2, R, T)
    camera.K = K
    camera.update_M_M_pinv_by_K_RT()

    global detector
    detector = od.ObjectorDetector(CKPT_FILE_PATH, LABEL_FILE_PATH, NUM_CLASS)


frames_cache = None


def main(imagesL, imagesR, confidence=0.2):
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

    global frames_cache
    frames_cache = frame_list

    world = wd.World()
    projections = fm.collect_projections_from_frames_by_confidence(frame_list, confidence=confidence)
    world.add_projections(projections)
    world.unify_objects_projection_get_object()

    # CHECK_IMAGE_PATH_FORMAT = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files/check_image/image{}.jpg'
    # images = fm.object_depth_detection_check_plot(frame_list, confidence=confidence)
    # for i, image in enumerate(images):
    #     print('saving image: ', i)
    #     cv2.imwrite(CHECK_IMAGE_PATH_FORMAT.format(i), image)
    return world.get_json()


def get_objects_again_confidence(confidence):
    global frames_cache
    if frames_cache is not None:
        projections = fm.collect_projections_from_frames_by_confidence(frames_cache, confidence)
        world = wd.World()
        world.add_projections(projections)
        world.unify_objects_projection_get_object()
        return world.get_json()
    else:
        print('Warning: no frames generated')


def get_objects_again_confidence_rank(rank):
    global frames_cache
    if frames_cache is not None:
        projections = fm.collect_projections_from_frames_by_confidence_rank(frames_cache, rank)
        world = wd.World()
        world.add_projections(projections)
        world.unify_objects_projection_get_object()
        return world.get_json()
    else:
        print('Warning: no frames generated')


def convert_portrait(image):
    shape = image.shape
    if shape[0] < shape[1]:
        image = np.rot90(image, -1, axes=(0, 1))
    return image


def test():
    from video_process import video_process
    VIDEO_PATH_L = '/Users/zijunyan/Downloads/new test/left.mp4'
    VIDEO_PATH_R = '/Users/zijunyan/Downloads/new test/right.mp4'
    VIDEO_PATH_C = '/Users/zijunyan/Downloads/video_cal.mp4'
    FRAME_DIR_PATH = '../../data/image_files'
    STEP = 20

    # images_cal = video_process.capture_frames_from_video(VIDEO_PATH_C, 5)

    imagesL = video_process.capture_frames_from_video(VIDEO_PATH_L, STEP)
    imagesR = video_process.capture_frames_from_video(VIDEO_PATH_R, STEP)

    for i, image in enumerate(imagesL):
        imagesL[i] = convert_portrait(image)

    for i, image in enumerate(imagesR):
        imagesR[i] = convert_portrait(image)
    setup()
    json = main(imagesL, imagesR)
    print(json)
    json = get_objects_again_confidence(0.3)
    print(json)
    json = get_objects_again_confidence_rank(3)
    print(json)

    # fm.motion_check_plot(frames)
    # fm.object_depth_detection_check_plot()


# test()
