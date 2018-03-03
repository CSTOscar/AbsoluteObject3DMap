from detector.depth_detector import detect_depth, detect_depth_image_3D
from frame.frame import generate_raw_frame_chain_from_images
from camera.camera import Camera
import numpy as np
import cv2
from pyntcloud import PyntCloud
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
from laspy.file import File


# def write_xyz(fout, coords, title="", atomtypes=("A",)):
#     """ write a xyz file from file handle
#     Writes coordinates in xyz format. It uses atomtypes as names. The list is
#     cycled if it contains less entries than there are coordinates,
#     One can also directly write xyz data which was generated with read_xyz.
#     # >>> xx = read_xyz("in.xyz")
#     # >>> write_xyz(open("out.xyz", "w"), *xx)
#     Parameters
#     ----------
#     fout : an open file
#     coords : np.array
#         array of coordinates
#     title : title section, optional
#         title for xyz file
#     atomtypes : iteratable
#         list of atomtypes.
#     See Also
#     --------
#     read_xyz
#     """
#     fout.write("%d\n%s\n" % (coords.size / 3, title))
#     for x, atomtype in zip(coords.reshape(-1, 3), cycle(atomtypes)):
#         fout.write("%s %.18g %.18g %.18g\n" % (atomtype, x[0], x[1], x[2]))


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
        # image_3d = np.array(detect_depth_image_3D(frame_input[0]))
        # reshaped_img = image_3d.reshape((image_3d.shape[0] * image_3d.shape[1], 3))

        # depth map
        depth_map  = detect_depth(frame_input[0])


        # # turn them into point cloud
        # point_list = pd.DataFrame(image_3d.reshape((image_3d.shape[0] * image_3d.shape[1], 3)))
        # point_list.columns = ['x', 'y', 'z']
        #
        #
        #
        #
        # # with open('test.xyz', 'w') as f:
        # #     write_xyz(f, reshaped_img, ('A','B'))
        #
        #
        # # cloud = PyntCloud(points=point_list)
        # # cloud.to_file("test1.ply")
        #
        # print(point_list)





