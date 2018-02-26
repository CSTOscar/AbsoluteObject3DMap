#! /usr/bin/env python3

import numpy as np
import math
import random
import scipy
import cv2 as cv

iden = np.zeros((4,4))
iden[0,0] = 1.0
iden[1,1] = 1.0
iden[2,2] = 1.0
iden[3,3] = 1.0
iden3 = iden[0:3,0:3]
se3_x = np.zeros((4,4))
se3_x[0,3] = 1.0
se3_y = np.zeros((4,4))
se3_y[1,3] = 1.0
se3_z = np.zeros((4,4))
se3_z[2,3] = 1.0
se3_xcos = np.zeros((4,4))
se3_xcos[1,1] = 1.0
se3_xcos[2,2] = 1.0
se3_xsin = np.zeros((4,4))
se3_xsin[1,2] = -1.0
se3_xsin[2,1] = 1.0
se3_ycos = np.zeros((4,4))
se3_ycos[0,0] = 1.0
se3_ycos[2,2] = 1.0
se3_ysin = np.zeros((4,4))
se3_ysin[0,2] = 1.0
se3_ysin[2,0] = -1.0
se3_zcos = np.zeros((4,4))
se3_zcos[0,0] = 1.0
se3_zcos[1,1] = 1.0
se3_zsin = np.zeros((4,4))
se3_zsin[0,1] = -1.0
se3_zsin[1,0] = 1.0

# Lie algebra to Lie group conversion of se(3) using Rodriguez formula for so(3)

def se3exp(xi):
    B = xi[3] * se3_xsin + xi[4] * se3_ysin + xi[5] * se3_zsin
    theta = np.linalg.norm(xi[3:])
    if theta == 0:
        return iden + xi[0] * se3_x + xi[1] * se3_y + xi[2] * se3_z
    return (iden + xi[0] * se3_x + xi[1] * se3_y + xi[2] * se3_z + 
            B * math.sin(theta) / theta + 
            (B @ B) * (1.0 - math.cos(theta)) / theta / theta)

def se3log(M):
    R = M[0:3,0:3]
    ct = ((R.trace() - 1.0) / 2.0)
    theta = math.acos((R.trace() - 1.0) / 2.0)
    if ct < -0.7: # more numerically stable method when \theta is near \pi
        RmI = R - iden3
        BS = (RmI + RmI.transpose()) * (theta * theta / 4.0 / (1-ct))
        ab = BS[2,2]
        bc = BS[0,0]
        ca = BS[1,1]
        return np.array([M[0,3],M[1,3],M[2,3],
            math.sqrt(bc-ab-ca), math.sqrt(ca-bc-ab), math.sqrt(ab-ca-bc)])
    else:
        st = math.sqrt(1.0 - ct * ct)
        mult = (0.5 if st < 0.00000001 else (theta / st / 2.0))
        B = (R - R.transpose()) * mult
        return np.array([M[0,3],M[1,3],M[2,3],B[2,1],B[0,2],B[1,0]])

def imgval(I, p):
    w, h = I.shape
    x = int(p[0] / p[2] + w/2)
    y = int(p[1] / p[2] + h/2)
    if x < 0 or x >= w or y < 0 or y >= h:
        return math.nan
    return int(I[x, y])

def imggrad(I, p, delta):
    w, h = I.shape
    x = p[0] / p[2]
    y = p[1] / p[2]
    d0 = (delta[0] - (x - w/2) * delta[2]) / p[2]
    d1 = (delta[1] - (y - h/2) * delta[1]) / p[2]
    x = int(x + w/2)
    y = int(y + h/2)
    #print(x, y, w, h)
    if x <= 0 or x >= w-1 or y <= 0 or y >= h-1:
        return 0.0
    if math.fabs(d0) < 0.00001 and math.fabs(d1) < 0.00001:
        return 0.0
    t = math.atan2(d1, d0)
    #print(t)
    #dlsq = (d0 * d0 + d1 * d1) / (p[2] * p[2])
    dlsq = d0 * d0 + d1 * d1
    if t < 0:
        if t < -math.pi / 2:
            if t < -math.pi * 3 / 4:
                u = d1 / d0
                #print(t)
                return ((1.0 - u) * I[x-1, y] + u * I[x-1, y-1])/math.sqrt((1+u*u)*dlsq)
            else:
                u = d0 / d1
                #print(u)
                return ((1.0 - u) * I[x, y-1] + u * I[x-1, y-1])/math.sqrt((1+u*u)*dlsq)
        else:
            if t < -math.pi / 4:
                u = d0 / d1
                #print(u)
                return ((1.0 + u) * I[x, y-1] - u * I[x+1, y-1])/math.sqrt((1+u*u)*dlsq)
            else:
                u = d1 / d0
                #print(t)
                return ((1.0 + u) * I[x+1, y] - u * I[x+1, y-1])/math.sqrt((1+u*u)*dlsq)
    else:
        if t < math.pi / 2:
            if t < math.pi / 4:
                #print(t)
                u = d1 / d0
                #print(u)
                return ((1.0 - u) * I[x+1, y] + u * I[x+1, y+1])/math.sqrt((1+u*u)*dlsq)
            else:
                u = d0 / d1
                #print(u)
                return ((1.0 - u) * I[x, y+1] + u * I[x+1, y+1])/math.sqrt((1+u*u)*dlsq)
        else:
            if t < math.pi * 3 / 4:
                u = d0 / d1
                #print(u)
                return ((1.0 + u) * I[x, y+1] - u * I[x-1, y+1])/math.sqrt((1+u*u)*dlsq)
            else:
                u = d1 / d0
                #print(t)
                return ((1.0 + u) * I[x-1, y] - u * I[x-1, y+1])/math.sqrt((1+u*u)*dlsq)

def slamResiduals(Ii, Ij, p, xi):
    four, k = p.shape
    r = 0
    xim = se3exp(xi)
    m = xim @ p
    bad = 0

    for i in range(k):
        #print(imgval(Ii, p[:,i]) - imgval(Ij, m[:,i]))
        ri = imgval(Ii, p[:, i]) - imgval(Ij, m[:, i])
        if math.isnan(ri):
            bad += 1
        else:
            r += ri * ri
    if bad > k/2:
        return math.inf

    print("bad: ", bad, "/", k)

    return r / (k - bad)

def slamStepIterate(Ii, Ij, p, xi, maxIter = 0):
    if maxIter <= 0:
        return xi

    xim = se3exp(xi) # matrix form of transformation

    four, k = p.shape # number of keyframes
    J = np.zeros((k, 6)) # Jacobian
    r = np.zeros((k)) # Residual vector

    m = xim @ p # transformed world points of keyframes


    # calculate Jacobian
    for j, g in zip(range(6),[se3_x, se3_y, se3_z, se3_xsin, se3_ysin, se3_zsin]):
        gm = g @ m
        for i in range(k):
            J[i, j] = imggrad(Ij, m[:, i], gm[:, i])

    # calculate residuals

    #print(p.shape)
    for i in range(k):
        r[i] = imgval(Ii, p[:, i]) - imgval(Ij, m[:, i])

    # perform a step of Gauss-Newton
    JT = J.transpose()
    print("STEP ")
    try:
        dxi = np.linalg.inv(JT @ J) @ JT @ r * -1.0
    except (np.linalg.linalg.LinAlgError):
        return xi
    #print(K.shape)
    #print(m.shape)
    return slamStepIterate(Ii, Ij, m, se3log(se3exp(dxi) @ xim), maxIter - 1)



    
    
    


def slamStep(pix0, dm0, pix1, k=1000):
    #print(pix0.shape)
    #print(dm0.shape)
    h, w = pix0.shape
    xi0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # TODO find keyframes using some sophisticated method
    K=np.zeros((k,3))
    for i in range(k):
        K[i,0] = int(random.randrange(int(h * -0.4), int(h * 0.4)))
        K[i,1] = int(random.randrange(int(w * -0.4), int(w * 0.4)))
        #print(K[i,0])
        #print(K[i,1])
        K[i,2] = dm0[int(K[i,0]), int(K[i,1])]

    p = np.array([K[:,0]*K[:,2],K[:,1]*K[:,2],K[:,2], np.ones(k)]) # world points of keyframes
    #return slamStepIterate(pix0, pix1, p, xi0, 100)
    xi = scipy.optimize.fmin(lambda xi: slamResiduals(pix0, pix1, p, xi), xi0)






def getMatches(px0, px1, lowe=0.4):
    sift = cv.xfeatures2d.SIFT_create()
    kp0, des0 = sift.detectAndCompute(px0, None)
    kp1, des1 = sift.detectAndCompute(px1, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des0, des1, k=2)

    good = []
    pts1 = []
    pts0 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < lowe * n.distance:
            good.append(m)
            pts1.append(kp1[m.trainIdx].pt)
            pts0.append(kp0[m.queryIdx].pt)

    return good, pts0, pts1





def slamStepIf3(img0, img1, img2, trans01, f):
    B = np.linalg.norm(trans01)
    px0 = cv.imread(img0, 0)
    px1 = cv.imread(img1, 0)
    px2 = cv.imread(img2, 0)

    good, pts0, pts1 = getMatches(px0, px1)

    pts0 = np.array(pts0)
    pts1 = np.array(pts1)
    disp = np.linalg.norm(pts1 - pts0, axis=1)
    depth = B * f / disp
    p = np.array([pts0[:,0] * depth, pts0[:,1] * depth, depth, np.ones(depth.shape)])

    #return slamStepIterate(pix0, pix1, p, xi0, 100)
    xi0 = np.array([0,0,0,0,0,0])

    xi = scipy.optimize.fmin(lambda xi: slamResiduals(px0, px1, p, xi), xi0)

    return xi



