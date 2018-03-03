from detector.depth_detector import detect_depth
from frame.frame import generate_raw_frame_chain_from_images
from camera.camera import Camera
import numpy as np
import cv2
from pyntcloud import PyntCloud
import pandas as pd


if __name__ == '__main__':

    for i in range(1, 2):
        image_left = cv2.imread('images/image_left/MouldShotTest%s_l.JPG' % (i))
        image_right = cv2.imread('images/image_right/MouldShotTest%s_r.JPG' % (i))
        R = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        T = np.asmatrix([[0], [0], [0]])
        camera_input = Camera(86500, 86500, 0.035, 2448 / 2, 3264 / 2, R, T)
        image_cali = cv2.imread('images/camera_cali_image_bai.JPG')
        camera_input.calibrate_by_images_and_grid_length(image_cali, 0.014)
        frame_input = generate_raw_frame_chain_from_images([image_left], [image_right], camera_input)
        # get 3d points
        image_3d = np.array(detect_depth(frame_input[0]))
        # turn them into point cloud
        point_list = pd.DataFrame(image_3d.reshape((image_3d.shape[0] * image_3d.shape[1], 3)))
        point_list.columns = ['x', 'y', 'z']
        # cloud = PyntCloud(points=point_list)
        # cloud.to_file("test1.ply")

        print(point_list)