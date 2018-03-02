import cv2
import numpy as np


class DepthDetector:

    @staticmethod
    def detect_depth(frame):


        imageL = frame.imageL
        imageR = frame.imageR

        # disparity settings
        window_size = 5
        min_disp = 1
        num_disp = 129 - min_disp

        matcher_left = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=15,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=200,
            speckleRange=2,
            # preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        matcher_right = cv2.ximgproc.createRightMatcher(matcher_left=matcher_left)

        # filter parameters
        lambda_ = 80000
        sigma = 1.2
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher_left)
        wls_filter.setLambda(_lambda=lambda_)
        wls_filter.setSigmaColor(sigma)

        imageL_bw = cv2.cvtColor(imageL, cv2.COLOR_RGB2GRAY)
        imageR_bw = cv2.cvtColor(imageR, cv2.COLOR_RGB2GRAY)

        # stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)


        # cv2.imshow('left eye', image_left)

        # compute disparity
        # disparity = matcher_left.compute(imageL, imageR).astype(np.float32)
        # disparity = (disparity - min_disp) / num_disp

        print('computing disparity...')
        displ = matcher_left.compute(imageL_bw, imageR_bw)
        dispr = matcher_right.compute(imageR_bw, imageL_bw)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        # filter the image
        filteredImg = wls_filter.filter(displ, imageL, None, dispr)
        # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)

        Q_matrix = None

        # parameter needed for the reprojection
        cv2.stereoRectify(frame.camera.K, [0,0,0,0], frame.camera.K, [0,0,0,0], image_left.size, frame.camera.R, frame.camera.T, Q=Q_matrix)


        cv2.reprojectImageTo3D(filteredImg, Q=Q_matrix)

        # self.K = intrinsic_matrix
        # self.R = rotation_matrix
        # self.T = transformation_vector
        # # print(self.K)
        # # print(self.R)
        # # print(self.T)
        # self.RT = Camera.generate_RT_from_R_T(self.R, self.T)
        # self.M = self.K @ self.RT
        # self.M_pinv = np.linalg.pinv(self.M)
        # return True

        # print(np.min(filteredImg), " ", np.max(filteredImg), " ", np.average(filteredImg))

        filteredImg = np.uint8(filteredImg)

        # morphology settings
        # kernel = np.ones((3, 3), np.uint8)
        # apply morphological transformation
        # morphology = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)

        # show images

        cv2.namedWindow('left eye', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('left eye', 1200, 900)
        cv2.imshow('left eye', imageL)

        cv2.namedWindow('right eye', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('right eye', 1200, 900)
        cv2.imshow('right eye', imageR)

        cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('disparity', 1200, 900)
        cv2.imshow('disparity', filteredImg)


        # cv2.namedWindow('morphology', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('morphology', 1200, 900)
        # cv2.imshow('morphology', morphology)

        print("done")
        cv2.waitKey(0)

        return


if __name__ == '__main__':

    for i in range(1, 4):
        image_left = cv2.imread('images/image_left/MouldShotTest%s_l.JPG' % (i))
        image_right = cv2.imread('images/image_right/MouldShotTest%s_r.JPG' % (i))
        DepthDetector.detect_depth(image_left, image_right)

