import cv2

IMAGE_PATH = '/Users/zijunyan/Desktop/Oscar/AbsoluteObject3DMap/data/image_files/image_l/MouldShotTest2_l.JPG'

image = cv2.imread(IMAGE_PATH)

print(type(image))
print(image.shape)

cv2.circle(image, (1000, 1500), 5, (0, 0, 255), thickness=10)
cv2.rectangle(image,(200,))

cv2.imshow('test', image)
cv2.waitKey()
