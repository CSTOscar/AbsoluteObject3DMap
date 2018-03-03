import numpy as np
import cv2 as cv
import math
import scipy.optimize
import matplotlib.pyplot as plt
import random


def debugPlot(p, xi, px0, px1, px2, K, pts1, pts3):
    R,_ = cv.Rodrigues(xi[3:])
    pi = K @ p
    pj = K @ (R @ p + xi[:3, np.newaxis])

    pts0 = (pi[:2, :] / pi[2, :]).transpose()
    pts2 = (pj[:2, :] / pj[2, :]).transpose()

    nn, two = pts0.shape
    print(nn)
    print(px0.shape)

    px0p = px0
    px1p = px1
    px2p = px2
    px3p = px2
    for i in range(nn):
        colour = (random.randint(60, 256), random.randint(0, 200), random.randint(20, 200))
        j = i  # random.randint(0, nn-1)
        # colour = (depth[j] * 50, 0, 256 - depth[j] * 50)
        # colour = hsv2rgb(depth[j] * 0.22, 1, 1)
        # print(tuple(pts0[j,:].astype(int).tolist()), tuple(pts1[j,:].astype(int).tolist()), disp[j], depth[j])
        px0p = cv.circle(px0p, tuple(pts0[j, :].astype(int).tolist()), 10, colour, -1)
        px1p = cv.circle(px1p, tuple(pts1[j, :].astype(int).tolist()), 10, colour, -1)
        px2p = cv.circle(px2p, tuple(pts2[j, :].astype(int).tolist()), 10, colour, -1)
        px3p = cv.circle(px3p, tuple(pts3[j, :].astype(int).tolist()), 10, colour, -1)

    plt.subplot(141), plt.imshow(px0p)
    plt.subplot(142), plt.imshow(px1p)
    plt.subplot(143), plt.imshow(px2p)
    plt.subplot(144), plt.imshow(px3p)
    plt.show()


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

    kp2 = prevframe.kp_left
    kp1 = frame.kp_right
    kp0 = frame.kp_left

    if len(kp0) < 20 or len(kp1) < 20 or len(kp2) < 20:
        raise MotionDetectionFailed("Not enough keypoints")

    # Find matches between the two images by the means of some dark magic
    flann = cv.FlannBasedMatcher(index_params_sift, search_params)
    matches01 = flann.knnMatch(frame.des_right, frame.des_left, k=2)
    matches02 = flann.knnMatch(prevframe.des_left, frame.des_left, k=2)

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
    print(xi)
    debugPlot(p, xi, frame.imageL, frame.imageR, prevframe.imageL, K, pts1, pts2)

    return xi
