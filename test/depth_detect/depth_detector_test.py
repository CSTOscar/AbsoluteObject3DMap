from matplotlib import pyplot

from detector import depth_detector_deprecated
from video_process import video_process

CKPT_FILE_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/model_files/fcrn_DepthPrediction/nyu_fcrn_checkpoint/NYU_FCRN.ckpt'


def test1():

    VIDEO_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/video_files/video0.mp4'
    FRAME_DIR_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files'

    STEP = 10

    images = video_process.capture_frames_from_video(VIDEO_PATH, STEP)

    print(images[0].shape)
    shape = images[0].shape
    detc = depth_detector_deprecated.DepthDetector(CKPT_FILE_PATH, *shape[:2])

    depth = detc.detect_depth_for_image(images[0])


    pyplot.imshow(images[0])
    pyplot.show()

    print(depth.shape)
    pyplot.imshow(depth)
    pyplot.show()

