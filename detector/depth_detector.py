import cv2
import numpy as np



def detect_depth(frame):

    imageL = frame.imageL
    imageR = frame.imageR

    # disparity settings
    window_size = 5
    min_disp = 1
    num_disp = 129 - min_disp

    matcher_left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=15,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=200,
        speckleRange=2,
        # preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    matcher_right = cv2.ximgproc.createRightMatcher(matcher_left=matcher_left)

    # filter parameters
    lambda_ = 80000
    sigma = 1.2
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher_left)
    wls_filter.setLambda(_lambda=lambda_)
    wls_filter.setSigmaColor(sigma)

    imageL_bw = cv2.cvtColor(imageL, cv2.COLOR_RGB2GRAY)
    imageR_bw = cv2.cvtColor(imageR, cv2.COLOR_RGB2GRAY)

    # stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)


    # cv2.imshow('left eye', image_left)

    # compute disparity
    # disparity = matcher_left.compute(imageL, imageR).astype(np.float32)
    # disparity = (disparity - min_disp) / num_disp

    print('computing disparity...')
    displ = matcher_left.compute(imageL_bw, imageR_bw)
    dispr = matcher_right.compute(imageR_bw, imageL_bw)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    # filter the image
    filteredImg = wls_filter.filter(displ, imageL, None, dispr)
    # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)

    Q_matrix = None

    # parameter needed for the reprojection
    img_shape = imageL.shape[:2]
    rotation_matrix_input = np.float64(frame.camera.R)
    translation_matrix_input = np.float64(frame.camera.T)
    camera_intrinsics = np.delete(frame.camera.K, 3, 1)
    camera_distortion = np.array(frame.camera.Dist)

    Q_matrix = cv2.stereoRectify(camera_intrinsics, camera_distortion, camera_intrinsics, camera_distortion, img_shape,
                      rotation_matrix_input,
                      translation_matrix_input)[4]

    image_3d = cv2.reprojectImageTo3D(np.array(filteredImg), Q=Q_matrix)

    # print(np.min(filteredImg), " ", np.max(filteredImg), " ", np.average(filteredImg))

    filteredImg = np.uint8(filteredImg)

    # morphology settings
    # kernel = np.ones((3, 3), np.uint8)
    # apply morphological transformation
    # morphology = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)

    # show images

    # cv2.namedWindow('left eye', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('left eye', 1200, 900)
    # cv2.imshow('left eye', imageL)
    #
    # cv2.namedWindow('right eye', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('right eye', 1200, 900)
    # cv2.imshow('right eye', imageR)
    #
    # cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('disparity', 1200, 900)
    # cv2.imshow('disparity', filteredImg)


    # cv2.namedWindow('morphology', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('morphology', 1200, 900)
    # cv2.imshow('morphology', morphology)

    depth_map = np.delete(image_3d, [0,1], axis=2)
    # print(depth_map)
    # depth_map = np.nan_to_num(depth_map)
    # depth_map *= 255.0 / np.max(depth_map)
    # depth_map = np.uint8(depth_map)
    #
    # cv2.namedWindow('depth map', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('depth map', 1200, 900)
    # cv2.imshow('depth map', depth_map)

    print("done")
    cv2.waitKey(0)

    return depth_map



def detect_depth_image_3D(frame):

    imageL = frame.imageL
    imageR = frame.imageR

    # disparity settings
    window_size = 5
    min_disp = 1
    num_disp = 129 - min_disp

    matcher_left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=15,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=200,
        speckleRange=2,
        # preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    matcher_right = cv2.ximgproc.createRightMatcher(matcher_left=matcher_left)

    # filter parameters
    lambda_ = 80000
    sigma = 1.2
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher_left)
    wls_filter.setLambda(_lambda=lambda_)
    wls_filter.setSigmaColor(sigma)

    imageL_bw = cv2.cvtColor(imageL, cv2.COLOR_RGB2GRAY)
    imageR_bw = cv2.cvtColor(imageR, cv2.COLOR_RGB2GRAY)

    # stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)


    # cv2.imshow('left eye', image_left)

    # compute disparity
    # disparity = matcher_left.compute(imageL, imageR).astype(np.float32)
    # disparity = (disparity - min_disp) / num_disp

    print('computing disparity...')
    displ = matcher_left.compute(imageL_bw, imageR_bw)
    dispr = matcher_right.compute(imageR_bw, imageL_bw)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    # filter the image
    filteredImg = wls_filter.filter(displ, imageL, None, dispr)
    # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)

    Q_matrix = None

    # parameter needed for the reprojection
    img_shape = imageL.shape[:2]
    rotation_matrix_input = np.float64(frame.camera.R)
    translation_matrix_input = np.float64(frame.camera.T)
    camera_intrinsics = np.delete(frame.camera.K, 3, 1)
    camera_distortion = np.array(frame.camera.Dist)

    Q_matrix = cv2.stereoRectify(camera_intrinsics, camera_distortion, camera_intrinsics, camera_distortion, img_shape,
                      rotation_matrix_input,
                      translation_matrix_input)[4]

    image_3d = cv2.reprojectImageTo3D(np.array(filteredImg), Q=Q_matrix)

    # print(np.min(filteredImg), " ", np.max(filteredImg), " ", np.average(filteredImg))

    filteredImg = np.uint8(filteredImg)

    # morphology settings
    # kernel = np.ones((3, 3), np.uint8)
    # apply morphological transformation
    # morphology = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)

    # show images

    # cv2.namedWindow('left eye', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('left eye', 1200, 900)
    # cv2.imshow('left eye', imageL)
    #
    # cv2.namedWindow('right eye', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('right eye', 1200, 900)
    # cv2.imshow('right eye', imageR)
    #
    # cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('disparity', 1200, 900)
    # cv2.imshow('disparity', filteredImg)


    # cv2.namedWindow('morphology', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('morphology', 1200, 900)
    # cv2.imshow('morphology', morphology)


    print("done")
    cv2.waitKey(0)

    return image_3d

