import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from pose_vector_to_transformation_matrix import \
    pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized


def main():
    pass

    # load camera poses

    # each row i of matrix 'poses' contains the transformations that transforms
    # points expressed in the world frame to
    # points expressed in the camera frame

    pose_vectors = np.loadtxt('../data/poses.txt')

    # define 3D corner positions
    # [Nx3] matrix containing the corners of the checkerboard as 3D points
    # (X,Y,Z), expressed in the world coordinate system

    square_size = 0.04  # [m]
    num_corners_x = 9
    num_corners_y = 6
    num_corners = num_corners_x * num_corners_y

    X, Y = np.meshgrid(np.arange(num_corners_x), np.arange(num_corners_y))
    p_W_corners = square_size * np.stack([X, Y],
                                         axis=-1).reshape([num_corners, 2])

    # add z coordinate, set z=0
    p_W_corners = np.concatenate(
        [p_W_corners, np.zeros([num_corners, 1])], axis=-1)

    # load camera intrinsics
    K = np.loadtxt('../data/K.txt')  # calibration matrix[3x3]
    D = np.loadtxt('../data/D.txt')  # distortion coefficients[2x1]

    # load one image with a given index
    img_index = 1
    img = cv2.imread('../data/images/img_{0:04d}.jpg'.format(img_index),
                     cv2.IMREAD_GRAYSCALE)


    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame

    T_C_W = pose_vector_to_transformation_matrix(pose_vectors[img_index, :])


    # transform 3d points from world to current camera pose
    p_C_corners = np.matmul(T_C_W[None, :, :],
                            np.concatenate([p_W_corners,
                                            np.ones([num_corners, 1])],
                                           axis=-1)[:, :, None]).squeeze(-1)
    p_C_corners = p_C_corners[:, :3]

    projected_pts = project_points(p_C_corners, K, D)
    plt.imshow(img, cmap='gray')
    plt.plot(projected_pts[:, 0], projected_pts[:, 1], 'r+')
    plt.show()

    # undistort image with bilinear interpolation
    start_t = time.time()
    img_undistorted = undistort_image(img, K, D, bilinear_interpolation=True)
    print('Undistortion with bilinear interpolation completed in {}'.format(
        time.time() - start_t))

    # vectorized undistortion without bilinear interpolation
    start_t = time.time()
    img_undistorted_vectorized = undistort_image_vectorized(img, K, D)
    print('Vectorized undistortion completed in {}'.format(
        time.time() - start_t))
    
    plt.clf()
    plt.close()
    fig, axs = plt.subplots(2)
    axs[0].imshow(img_undistorted, cmap='gray')
    axs[0].set_axis_off()
    axs[0].set_title('With bilinear interpolation')
    axs[1].imshow(img_undistorted_vectorized, cmap='gray')
    axs[1].set_axis_off()
    axs[1].set_title('Without bilinear interpolation')
    plt.show()

    # calculate the cube points to then draw the image
    offset_x = 0.04 * 3
    offset_y = 0.04
    s = 2 * 0.04

    X, Y, Z = np.meshgrid(np.arange(2), np.arange(2), np.arange(-1, 1))
    p_W_cube = np.stack([offset_x + X.flatten()*s,
                         offset_y + Y.flatten()*s,
                         Z.flatten()*s,
                         np.ones([8])], axis=-1)

    p_C_cube = np.matmul(T_C_W[None, :, :], p_W_cube[:, :, None]).squeeze(-1)
    p_C_cube = p_C_cube[:, :3]

    cube_pts = project_points(p_C_cube, K, np.zeros([4, 1]))
    
    # Plot the cube
    plt.clf()
    plt.close()
    plt.imshow(img_undistorted, cmap='gray')

    lw = 3

    # base layer of the cube
    plt.plot(cube_pts[[1, 3, 7, 5, 1], 0],
             cube_pts[[1, 3, 7, 5, 1], 1],
             'r-',
             linewidth=lw)

    # top layer of the cube
    plt.plot(cube_pts[[0, 2, 6, 4, 0], 0],
             cube_pts[[0, 2, 6, 4, 0], 1],
             'r-',
             linewidth=lw)

    # vertical lines
    plt.plot(cube_pts[[0, 1], 0], cube_pts[[0, 1], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[2, 3], 0], cube_pts[[2, 3], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[4, 5], 0], cube_pts[[4, 5], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[6, 7], 0], cube_pts[[6, 7], 1], 'r-', linewidth=lw)

    plt.show()


if __name__ == "__main__":
    main()
