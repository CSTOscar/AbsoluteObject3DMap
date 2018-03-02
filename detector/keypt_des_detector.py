import cv2 as cv

def detect_keypt_des(frame):
    sift = cv.xfeatures2d.SIFT_create()

    frame.kp_left, frame.des_left = sift.detectAndCompute(frame.imageL, None)
    frame.kp_right, frame.des_right = sift.detectAndCompute(frame.imageR, None)

def detect_keypt_des_orb(frame):
    orb = cv.ORB_create()

    frame.kp_left, frame.des_left = orb.detectAndCompute(frame.imageL, None)
    frame.kp_right, frame.des_right = orb.detectAndCompute(frame.imageR, None)

