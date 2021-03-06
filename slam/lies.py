#! /usr/bin/env python3

import numpy as np
import math

iden = np.identity(4)
iden3 = np.identity(3)
so3_xsin = np.array([[ 0.,  0.,  0.],
       [ 0.,  0., -1.],
       [ 0.,  1.,  0.]])
so3_ysin = np.array([[ 0.,  0.,  1.],
       [ 0.,  0.,  0.],
       [-1.,  0.,  0.]])
so3_zsin = np.array([[ 0., -1.,  0.],
       [ 1.,  0.,  0.],
       [ 0.,  0.,  0.]])
se3_x = np.array([[0., 0., 0., 1.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
se3_y = np.array([[0., 0., 0., 0.],
       [0., 0., 0., 1.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
se3_z = np.array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 1.],
       [0., 0., 0., 0.]])
se3_xsin = np.array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
se3_ysin = np.array([[ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.],
       [-1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
se3_zsin = np.array([[ 0., -1.,  0.,  0.],
       [ 1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])

# Lie algebra to Lie group conversion of se(3) using Rodriguez formula for so(3)

def so3exp(xi):
    theta = np.linalg.norm(xi)
    if theta == 0:
        return iden3
    B = xi[0] * so3_xsin + xi[1] * so3_ysin + xi[2] * so3_zsin
    return (iden3 +
            B * math.sin(theta) / theta + 
            (B @ B) * (1.0 - math.cos(theta)) / theta / theta)

def so3log(R):
    ct = ((R.trace() - 1.0) / 2.0)
    theta = math.acos(ct)
    if ct < -0.7: # more numerically stable method when \theta is near \pi
        RmI = R - iden3
        BS = (RmI + RmI.transpose()) * (theta * theta / 4.0 / (1-ct))
        ab = BS[2,2]
        bc = BS[0,0]
        ca = BS[1,1]
        return np.array([
            math.sqrt(bc-ab-ca), math.sqrt(ca-bc-ab), math.sqrt(ab-ca-bc)])
    else:
        st = math.sqrt(1.0 - ct * ct)
        mult = (0.5 if st < 0.00000001 else (theta / st / 2.0))
        B = (R - R.transpose()) * mult
        return np.array([B[2,1],B[0,2],B[1,0]])


def se3exp(xi):
    theta = np.linalg.norm(xi[3:])
    if theta == 0:
        return iden + xi[0] * se3_x + xi[1] * se3_y + xi[2] * se3_z
    B = xi[3] * se3_xsin + xi[4] * se3_ysin + xi[5] * se3_zsin
    return (iden + xi[0] * se3_x + xi[1] * se3_y + xi[2] * se3_z + 
            B * math.sin(theta) / theta + 
            (B @ B) * (1.0 - math.cos(theta)) / theta / theta)

def se3log(M):
    R = M[0:3,0:3]
    ct = ((R.trace() - 1.0) / 2.0)
    theta = math.acos(ct)
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
