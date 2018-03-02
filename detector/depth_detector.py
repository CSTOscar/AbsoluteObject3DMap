import cv2
import numpy as np


def detect_depth(self, frame):
    return


class DepthDetector:
    def __init__(self):
        return

    def detect_depth(self, imageL, imageR):
        # disparity settings
        window_size = 5
        min_disp = 1
        num_disp = 129 - min_disp
        # stereo = cv2.StereoSGBM_create(
        #     minDisparity=min_disp,
        #     numDisparities=num_disp,
        #     # SADWindowSize=window_size,
        #     uniquenessRatio=10,
        #     speckleWindowSize=100,
        #     speckleRange=50,
        #     disp12MaxDiff=1,
        #     P1=8 * 3 * window_size ** 2,
        #     P2=32 * 3 * window_size ** 2,
        #     # fullDP=False
        #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        # )

        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=11,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=200,
            speckleRange=2,
            # preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

        # morphology settings
        kernel = np.ones((3, 3), np.uint8)

        # only process every third image (so as to speed up video)
        # if counter % 3 != 0: continue

        # load stereo image
        # filename = str(counter).zfill(4)

        image_left = cv2.imread('images/image_left/20180226test%s_l.JPG' % (i), 0)
        image_right = cv2.imread('images/image_right/20180226test%s_r.JPG' % (i), 0)
        # cv2.imshow('left eye', image_left)

        # compute disparity
        disparity = stereo.compute(image_left, image_right).astype(np.float32) / 4
        disparity = (disparity - min_disp) / num_disp

        # apply threshold
        # threshold = cv2.threshold(disparity, 0.6, 1.0, cv2.THRESH_BINARY)[1]

        # apply morphological transformation
        morphology = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)

        # show images

        cv2.namedWindow('left eye', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('left eye', 1200, 900)
        cv2.imshow('left eye', image_left)

        cv2.namedWindow('right eye', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('right eye', 1200, 900)
        cv2.imshow('right eye', image_right)

        cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('disparity', 1200, 900)
        cv2.imshow('disparity', disparity)

        # cv2.imshow('threshold', threshold)

        cv2.namedWindow('morphology', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('morphology', 1200, 900)
        cv2.imshow('morphology', morphology)

        print("done")
        cv2.waitKey(0)

        return
