import cv2

print(cv2.__version__)
vidcap = cv2.VideoCapture('cv_image/video0_converted.avi')
success, image = vidcap.read()
count = 0
success = True
while success:
    # cv2.imwrite("cv_image/frames/image{}.jpg".format(count), image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    print(vidcap.get(10))
    count += 1