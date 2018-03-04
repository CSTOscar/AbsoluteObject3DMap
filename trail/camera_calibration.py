import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
objp *= 0.024
print(objp)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('*.jpg')

fname = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/chess_board_image/IMG_1389.JPG'

img = cv2.imread(fname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

# If found, add object points, image points (after refining them)
print(ret)

print(corners)

if ret == True:
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(ret)
    print(mtx)
    print(dist)
    print(rvecs)
    print(tvecs)
    rotation = cv2.Rodrigues(rvecs[0])[0]
    print(rotation)
    Z1 = np.asmatrix([[0], [0], [1]])
    print(rotation @ Z1)
    cv2.imshow('img', img)
    cv2.waitKey()
    # cv2.waitKey(500)

# cv2.destroyAllWindows()
