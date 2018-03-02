import numpy as np
import cv2 as cv
import math
import scipy.optimize

class MotionDetectionFailed(Exception):
    pass

def detect_motion(frame):
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

    # Find matches between the two images by the means of some dark magic
    flann = cv.FlannBasedMatcher(index_params_sift, search_params)
    matches01 = flann.knnMatch(prevframe.des_right, prevframe.des_left, k=2)
    matches02 = flann.knnMatch(frame.des_left, prevframe.des_left, k=2)

    pts2 = []
    pts1 = []
    pts0 = []
    kp2 = frame.kp_left
    kp1 = prevframe.kp_right
    kp0 = prevframe.kp_left

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
        raise MotionDetectionFailed("Not enough matches")
    # we should already know this matrix? Wierdly, we get something different than last time.
    F01, mask = cv.findFundamentalMat(pts1, pts0, cv.FM_LMEDS)

    # We select only inlier points
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    k = pts0.shape[0]
    print(k)
    if k < 15:
        raise MotionDetectionFailed("Not enough matches")

    F02, mask = cv.findFundamentalMat(pts2, pts0, cv.FM_LMEDS)
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    k = pts0.shape[0]
    print(k)
    if k < 15:
        raise MotionDetectionFailed("Not enough matches")

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

    # world points
    p = L @ pts0hom * depth

    # Reference for residuals matching
    ref = pts2.transpose()

    xi0 = np.array([0., 0., 0., 0., 0., 0.])
    xi = scipy.optimize.fmin(lambda xi: slamResidualMatches(p, xi, K, ref), xi0)

    return xi
