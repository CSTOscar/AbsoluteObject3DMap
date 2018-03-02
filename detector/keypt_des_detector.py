import cv2 as cv


def detect_keypt_des(frame):
    sift = cv.xfeatures2d.SIFT_create()

    yield sift.detectAndCompute(frame.imageL, None)
    yield sift.detectAndCompute(frame.imageR, None)


def detect_keypt_des_orb(frame):
    orb = cv.ORB_create()

    yield orb.detectAndCompute(frame.imageL, None)
    yield orb.detectAndCompute(frame.imageR, None)
