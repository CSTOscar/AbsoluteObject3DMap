'''
Simple example of stereo image matching and point cloud generation.
Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

import numpy as np
import cv2

from camera.camera import Camera
from frame.frame import generate_raw_frame_chain_from_images

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


if __name__ == '__main__':
    print('loading images...')
    i = 2
    imgL = cv2.imread('images/image_left/20180226test%s_l.jpg' % (i))
    imgR = cv2.imread('images/image_right/20180226test%s_r.JPG' % (i))

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                            numDisparities=num_disp,
                            uniquenessRatio=10,
                            speckleWindowSize=100,
                            speckleRange=32,
                            disp12MaxDiff=1,
                            P1=8 * 3 * window_size ** 2,
                            P2=32 * 3 * window_size ** 2,
                            )

    print('computing disparity...')

    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...',)
    R = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = np.asmatrix([[0], [0], [0]])
    camera_input = Camera(86500, 86500, 0.035, 2448 / 2, 3264 / 2, R, T)
    image_cali = cv2.imread('images/camera_cali_image_bai.JPG')
    camera_input.calibrate_by_images_and_grid_length(image_cali, 0.014)

    frame_input = generate_raw_frame_chain_from_images([imgL], [imgR], camera_input)
    frame_input = frame_input[0]
    img_shape = imgL.shape[:2]
    rotation_matrix_input = np.float64(frame_input.camera.R)
    translation_matrix_input = np.float64(frame_input.camera.T)
    camera_intrinsics = np.delete(frame_input.camera.K, 3, 1)
    camera_distortion = np.array(frame_input.camera.Dist)

    Q_matrix = cv2.stereoRectify(camera_intrinsics, camera_distortion, camera_intrinsics, camera_distortion, img_shape,
                      rotation_matrix_input,
                      translation_matrix_input)[4]

    points = cv2.reprojectImageTo3D(disp, Q_matrix)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')


    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp - min_disp) / num_disp)
    cv2.waitKey()
    cv2.destroyAllWindows()