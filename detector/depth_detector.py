import cv2 as cv
import numpy as np
import math
import scipy.optimize


class DepthDetectionFailed(Exception):
    pass


def detect_depth(frame):
    prevframe = frame.prev_frame

    # Parameters for flann matching
    search_params = dict(checks=50)

    # For SIFT descriptors
    FLANN_INDEX_KDTREE = 1
    index_params_sift = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    # For ORB descriptors
    # FLANN_INDEX_LSH = 6
    # index_params_orb = dict(algorithm = FLANN_INDEX_LSH,
    #                   table_number = 6, # 12
    #                   key_size = 12,     # 20
    #                   multi_probe_level = 1) #2

    kp2 = prevframe.kp_left
    kp1 = frame.kp_right
    kp0 = frame.kp_left

    if len(kp0) < 20 or len(kp1) < 20 or len(kp2) < 20:
        raise DepthDetectionFailed("Not enough keypoints")

    # Find matches between the two images by the means of some dark magic
    try:
        flann = cv.FlannBasedMatcher(index_params_sift, search_params)
        matches01 = flann.knnMatch(frame.des_right, frame.des_left, k=2)
        matches02 = flann.knnMatch(prevframe.des_left, frame.des_left, k=2)
    except Exception:
        raise DepthDetectionFailed("Not enough matches")

    pts2 = []
    pts1 = []
    pts0 = []

    good01 = {}

    # ratio test as per Lowe's paper
    lowe = 0.9
    for i, (m, n) in enumerate(matches01):
        if m.distance < lowe * n.distance:
            p1 = kp1[m.queryIdx].pt
            p0 = kp0[m.trainIdx].pt
            if p0[0] - p1[0] >= 10:
                good01[m.trainIdx] = m

    # Finding matches between all three images.
    for i, (m, n) in enumerate(matches02):
        if m.distance < lowe * n.distance:
            if m.trainIdx in good01:
                pts0.append(kp0[m.trainIdx].pt)
                pts1.append(kp1[good01[m.trainIdx].queryIdx].pt)
                pts2.append(kp2[m.queryIdx].pt)

    B = 0.09
    K = np.array(frame.camera.K)[:3, :3]
    L = np.linalg.inv(K)

    pts0 = np.int32(pts0)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    print()
    k = pts0.shape[0]
    print(k)
    if k < 15:
        raise DepthDetectionFailed("Not enough matches")
    # we should already know this matrix? Wierdly, we get something different than last time.
    F01, mask = cv.findFundamentalMat(pts1, pts0, cv.FM_LMEDS)

    # We select only inlier points
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    k = pts0.shape[0]
    print(k)
    if k < 15:
        raise DepthDetectionFailed("Not enough matches")

    F02, mask = cv.findFundamentalMat(pts2, pts0, cv.FM_LMEDS)
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    k = pts0.shape[0]
    print(k)
    if k < 15:
        raise DepthDetectionFailed("Not enough matches")

    def slamResidualMatches(p, xi, K1, ref):
        # p = input world points
        # xi = movement of camera
        # K1 = camera matrix of second camera
        # ref = reference points on second camera in screen points
        R, _ = cv.Rodrigues(xi[3:])
        pj = K1 @ (R @ p + xi[:3, np.newaxis])
        if not (pj[2, :] > 0).all():
            return math.inf

        pjn = pj[:2, :] / pj[2, :]

        return np.linalg.norm(pjn - ref[:2])

    # Finding world points of keypoints again
    k, two = pts0.shape

    # image points with homogeneous coordinates
    pts0hom = np.array([pts0[:, 0], pts0[:, 1], np.ones((k))])
    pts1hom = np.array([pts1[:, 0], pts1[:, 1], np.ones((k))])

    # disparity and depth
    disp = ((L @ pts0hom) - (L @ pts1hom))[0, :]
    depth = B / disp

    print('disp', disp)
    print(pts0hom)
    print(pts1hom)

    pixel_points = pts1hom.T[:, :2].astype(dtype=np.int_).tolist()

    depths = []
    for i, pixel_point in enumerate(pixel_points):
        depths.append((pixel_point, depth[i]))

    return depths

    # world points
    # p = L @ pts0hom * depth

    # Reference for residuals matching
    # ref = pts2.transpose()
    #
    # xi0 = np.array([0., 0., 0., 0., 0., 0.])
    # xi = scipy.optimize.fmin(lambda xi: slamResidualMatches(p, xi, K, ref), xi0)
    #
    # return xi

#
# class DepthDetector:
#
#     @staticmethod
#     def detect_depth(frame):
#
#
#         imageL = frame.imageL
#         imageR = frame.imageR
#
#         # disparity settings
#         window_size = 5
#         min_disp = 1
#         num_disp = 129 - min_disp
#
#         matcher_left = cv2.StereoSGBM_create(
#             minDisparity=0,
#             numDisparities=64,
#             blockSize=15,
#             P1=8 * 3 * window_size ** 2,
#             P2=32 * 3 * window_size ** 2,
#             disp12MaxDiff=1,
#             uniquenessRatio=15,
#             speckleWindowSize=200,
#             speckleRange=2,
#             # preFilterCap=63,
#             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
#         )
#
#         matcher_right = cv2.ximgproc.createRightMatcher(matcher_left=matcher_left)
#
#         # filter parameters
#         lambda_ = 80000
#         sigma = 1.2
#         wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher_left)
#         wls_filter.setLambda(_lambda=lambda_)
#         wls_filter.setSigmaColor(sigma)
#
#         imageL_bw = cv2.cvtColor(imageL, cv2.COLOR_RGB2GRAY)
#         imageR_bw = cv2.cvtColor(imageR, cv2.COLOR_RGB2GRAY)
#
#         # stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
#
#
#         # cv2.imshow('left eye', image_left)
#
#         # compute disparity
#         # disparity = matcher_left.compute(imageL, imageR).astype(np.float32)
#         # disparity = (disparity - min_disp) / num_disp
#
#         print('computing disparity...')
#         displ = matcher_left.compute(imageL_bw, imageR_bw)
#         dispr = matcher_right.compute(imageR_bw, imageL_bw)  # .astype(np.float32)/16
#         displ = np.int16(displ)
#         dispr = np.int16(dispr)
#         # filter the image
#         filteredImg = wls_filter.filter(displ, imageL, None, dispr)
#         # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
#
#         Q_matrix = None
#
#         # parameter needed for the reprojection
#         cv2.stereoRectify(frame.camera.K, [0,0,0,0], frame.camera.K, [0,0,0,0], image_left.size, frame.camera.R, frame.camera.T, Q=Q_matrix)
#
#
#         cv2.reprojectImageTo3D(filteredImg, Q=Q_matrix)
#
#         # self.K = intrinsic_matrix
#         # self.R = rotation_matrix
#         # self.T = transformation_vector
#         # # print(self.K)
#         # # print(self.R)
#         # # print(self.T)
#         # self.RT = Camera.generate_RT_from_R_T(self.R, self.T)
#         # self.M = self.K @ self.RT
#         # self.M_pinv = np.linalg.pinv(self.M)
#         # return True
#
#         # print(np.min(filteredImg), " ", np.max(filteredImg), " ", np.average(filteredImg))
#
#         filteredImg = np.uint8(filteredImg)
#
#         # morphology settings
#         # kernel = np.ones((3, 3), np.uint8)
#         # apply morphological transformation
#         # morphology = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)
#
#         # show images
#
#         cv2.namedWindow('left eye', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('left eye', 1200, 900)
#         cv2.imshow('left eye', imageL)
#
#         cv2.namedWindow('right eye', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('right eye', 1200, 900)
#         cv2.imshow('right eye', imageR)
#
#         cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('disparity', 1200, 900)
#         cv2.imshow('disparity', filteredImg)
#
#
#         # cv2.namedWindow('morphology', cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow('morphology', 1200, 900)
#         # cv2.imshow('morphology', morphology)
#
#         print("done")
#         cv2.waitKey(0)
#
#         return
#
#
# if __name__ == '__main__':
#
#     for i in range(1, 4):
#         image_left = cv2.imread('images/image_left/MouldShotTest%s_l.JPG' % (i))
#         image_right = cv2.imread('images/image_right/MouldShotTest%s_r.JPG' % (i))
#         DepthDetector.detect_depth(image_left, image_right)




# import cv2
# import numpy as np
#
#
#
# def detect_depth(frame):
#
#     imageL = frame.imageL
#     imageR = frame.imageR
#
#     # disparity settings
#     window_size = 5
#     min_disp = 1
#     num_disp = 129 - min_disp
#
#     matcher_left = cv2.StereoSGBM_create(
#         minDisparity=0,
#         numDisparities=64,
#         blockSize=15,
#         P1=8 * 3 * window_size ** 2,
#         P2=32 * 3 * window_size ** 2,
#         disp12MaxDiff=1,
#         uniquenessRatio=15,
#         speckleWindowSize=200,
#         speckleRange=2,
#         # preFilterCap=63,
#         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
#     )
#
#     matcher_right = cv2.ximgproc.createRightMatcher(matcher_left=matcher_left)
#
#     # filter parameters
#     lambda_ = 80000
#     sigma = 1.2
#     wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher_left)
#     wls_filter.setLambda(_lambda=lambda_)
#     wls_filter.setSigmaColor(sigma)
#
#     imageL_bw = cv2.cvtColor(imageL, cv2.COLOR_RGB2GRAY)
#     imageR_bw = cv2.cvtColor(imageR, cv2.COLOR_RGB2GRAY)
#
#     # stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
#
#
#     # cv2.imshow('left eye', image_left)
#
#     # compute disparity
#     # disparity = matcher_left.compute(imageL, imageR).astype(np.float32)
#     # disparity = (disparity - min_disp) / num_disp
#
#     print('computing disparity...')
#     displ = matcher_left.compute(imageL_bw, imageR_bw)
#     dispr = matcher_right.compute(imageR_bw, imageL_bw)  # .astype(np.float32)/16
#     displ = np.int16(displ)
#     dispr = np.int16(dispr)
#     # filter the image
#     filteredImg = wls_filter.filter(displ, imageL, None, dispr)
#     # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
#
#     Q_matrix = None
#
#     # parameter needed for the reprojection
#     img_shape = imageL.shape[:2]
#     rotation_matrix_input = np.float64(frame.camera.R)
#     translation_matrix_input = np.float64(frame.camera.T)
#     camera_intrinsics = np.delete(frame.camera.K, 3, 1)
#     camera_distortion = np.array(frame.camera.Dist)
#
#     Q_matrix = cv2.stereoRectify(camera_intrinsics, camera_distortion, camera_intrinsics, camera_distortion, img_shape,
#                       rotation_matrix_input,
#                       translation_matrix_input)[4]
#
#     image_3d = cv2.reprojectImageTo3D(np.array(filteredImg), Q=Q_matrix)
#
#     # print(np.min(filteredImg), " ", np.max(filteredImg), " ", np.average(filteredImg))
#
#     filteredImg = np.uint8(filteredImg)
#
#     # morphology settings
#     # kernel = np.ones((3, 3), np.uint8)
#     # apply morphological transformation
#     # morphology = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)
#
#     # show images
#
#     cv2.namedWindow('left eye', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('left eye', 1200, 900)
#     cv2.imshow('left eye', imageL)
#
#     cv2.namedWindow('right eye', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('right eye', 1200, 900)
#     cv2.imshow('right eye', imageR)
#
#     cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('disparity', 1200, 900)
#     cv2.imshow('disparity', filteredImg)
#
#     #
#     # cv2.namedWindow('morphology', cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('morphology', 1200, 900)
#     # cv2.imshow('morphology', morphology)
#
#     depth_map = np.delete(image_3d, [0,1], axis=2)
#
#     # change the inf to -inf before getting max
#     depth_map[depth_map == np.inf] = -np.inf
#     non_inf_max = np.max(depth_map)
#     # and replace all the inf with the non_inf_max
#     depth_map[depth_map == -np.inf] = non_inf_max
#
#     depth_map *= 255.0 / non_inf_max
#     depth_map = np.uint8(depth_map)
#
#     cv2.namedWindow('depth map', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('depth map', 1200, 900)
#     cv2.imshow('depth map', depth_map)
#
#     print("done")
#     cv2.waitKey(0)
#
#     return depth_map
#
#
#
# def detect_depth_image_3D(frame):
#
#     imageL = frame.imageL
#     imageR = frame.imageR
#
#     # disparity settings
#     window_size = 5
#     min_disp = 1
#     num_disp = 129 - min_disp
#
#     matcher_left = cv2.StereoSGBM_create(
#         minDisparity=0,
#         numDisparities=64,
#         blockSize=15,
#         P1=8 * 3 * window_size ** 2,
#         P2=32 * 3 * window_size ** 2,
#         disp12MaxDiff=1,
#         uniquenessRatio=15,
#         speckleWindowSize=200,
#         speckleRange=2,
#         # preFilterCap=63,
#         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
#     )
#
#     matcher_right = cv2.ximgproc.createRightMatcher(matcher_left=matcher_left)
#
#     # filter parameters
#     lambda_ = 80000
#     sigma = 1.2
#     wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher_left)
#     wls_filter.setLambda(_lambda=lambda_)
#     wls_filter.setSigmaColor(sigma)
#
#     imageL_bw = cv2.cvtColor(imageL, cv2.COLOR_RGB2GRAY)
#     imageR_bw = cv2.cvtColor(imageR, cv2.COLOR_RGB2GRAY)
#
#     # stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
#
#
#     # cv2.imshow('left eye', image_left)
#
#     # compute disparity
#     # disparity = matcher_left.compute(imageL, imageR).astype(np.float32)
#     # disparity = (disparity - min_disp) / num_disp
#
#     print('computing disparity...')
#     displ = matcher_left.compute(imageL_bw, imageR_bw)
#     dispr = matcher_right.compute(imageR_bw, imageL_bw)  # .astype(np.float32)/16
#     displ = np.int16(displ)
#     dispr = np.int16(dispr)
#     # filter the image
#     filteredImg = wls_filter.filter(displ, imageL, None, dispr)
#     # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
#
#     Q_matrix = None
#
#     # parameter needed for the reprojection
#     img_shape = imageL.shape[:2]
#     rotation_matrix_input = np.float64(frame.camera.R)
#     translation_matrix_input = np.float64(frame.camera.T)
#     camera_intrinsics = np.delete(frame.camera.K, 3, 1)
#     camera_distortion = np.array(frame.camera.Dist)
#
#     Q_matrix = cv2.stereoRectify(camera_intrinsics, camera_distortion, camera_intrinsics, camera_distortion, img_shape,
#                       rotation_matrix_input,
#                       translation_matrix_input)[4]
#
#     image_3d = cv2.reprojectImageTo3D(np.array(filteredImg), Q=Q_matrix)
#
#     # print(np.min(filteredImg), " ", np.max(filteredImg), " ", np.average(filteredImg))
#
#     filteredImg = np.uint8(filteredImg)
#
#     # morphology settings
#     # kernel = np.ones((3, 3), np.uint8)
#     # apply morphological transformation
#     # morphology = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)
#
#     # show images
#
#     # cv2.namedWindow('left eye', cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('left eye', 1200, 900)
#     # cv2.imshow('left eye', imageL)
#     #
#     # cv2.namedWindow('right eye', cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('right eye', 1200, 900)
#     # cv2.imshow('right eye', imageR)
#     #
#     # cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('disparity', 1200, 900)
#     # cv2.imshow('disparity', filteredImg)
#
#
#     # cv2.namedWindow('morphology', cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('morphology', 1200, 900)
#     # cv2.imshow('morphology', morphology)
#
#
#     print("done")
#     cv2.waitKey(0)
#
#     return image_3d
#
#
#
#
# def detect_depth_without_filter(frame):
#
#     imageL = frame.imageL
#     imageR = frame.imageR
#
#     # disparity settings
#     window_size = 5
#     min_disp = 1
#     num_disp = 129 - min_disp
#
#     matcher_left = cv2.StereoSGBM_create(
#         minDisparity=0,
#         numDisparities=64,
#         blockSize=15,
#         P1=8 * 3 * window_size ** 2,
#         P2=32 * 3 * window_size ** 2,
#         disp12MaxDiff=1,
#         uniquenessRatio=15,
#         speckleWindowSize=200,
#         speckleRange=2,
#         # preFilterCap=63,
#         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
#     )
#
#
#     imageL_bw = cv2.cvtColor(imageL, cv2.COLOR_RGB2GRAY)
#     imageR_bw = cv2.cvtColor(imageR, cv2.COLOR_RGB2GRAY)
#
#     # stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
#
#
#     # cv2.imshow('left eye', image_left)
#
#     # compute disparity
#     # disparity = matcher_left.compute(imageL, imageR).astype(np.float32)
#     # disparity = (disparity - min_disp) / num_disp
#
#     print('computing disparity...')
#     displ = matcher_left.compute(imageL_bw, imageR_bw)
#
#     Q_matrix = None
#
#     # parameter needed for the reprojection
#     img_shape = imageL.shape[:2]
#     rotation_matrix_input = np.float64(frame.camera.R)
#     translation_matrix_input = np.float64(frame.camera.T)
#     camera_intrinsics = np.delete(frame.camera.K, 3, 1)
#     camera_distortion = np.array(frame.camera.Dist)
#
#     Q_matrix = cv2.stereoRectify(camera_intrinsics, camera_distortion, camera_intrinsics, camera_distortion, img_shape,
#                       rotation_matrix_input,
#                       translation_matrix_input)[4]
#
#     image_3d = cv2.reprojectImageTo3D(np.array(displ), Q=Q_matrix)
#
#     # print(np.min(filteredImg), " ", np.max(filteredImg), " ", np.average(filteredImg))
#
#     filteredImg = np.uint8(displ)
#
#     # morphology settings
#     # kernel = np.ones((3, 3), np.uint8)
#     # apply morphological transformation
#     # morphology = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)
#
#     # show images
#
#     cv2.namedWindow('left eye', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('left eye', 1200, 900)
#     cv2.imshow('left eye', imageL)
#
#     cv2.namedWindow('right eye', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('right eye', 1200, 900)
#     cv2.imshow('right eye', imageR)
#
#     cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('disparity', 1200, 900)
#     cv2.imshow('disparity', filteredImg)
#
#     #
#     # cv2.namedWindow('morphology', cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('morphology', 1200, 900)
#     # cv2.imshow('morphology', morphology)
#
#     depth_map = np.delete(image_3d, [0,1], axis=2)
#
#     # change the inf to -inf before getting max
#     depth_map[depth_map == np.inf] = -np.inf
#     non_inf_max = np.max(depth_map)
#     # and replace all the inf with the non_inf_max
#     depth_map[depth_map == -np.inf] = non_inf_max
#
#     depth_map *= 200.0 / non_inf_max
#     depth_map = np.uint8(depth_map)
#
#     cv2.namedWindow('depth map', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('depth map', 1200, 900)
#     cv2.imshow('depth map', depth_map)
#
#     print("done")
#     cv2.waitKey(0)
#
#     return depth_map





