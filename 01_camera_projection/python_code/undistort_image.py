import math
import numpy as np

from distort_points import distort_points


def undistort_image(img: np.ndarray,
                    K: np.ndarray,
                    D: np.ndarray,
                    bilinear_interpolation: bool = False) -> np.ndarray:
    """
    Corrects an image for lens distortion.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)
        bilinear_interpolation: whether to use bilinear interpolation or not
    """
    pass
    height, width = img.shape

    undistorted_img = np.zeros([height, width])

    for x in range(width):
        for y in range(height):

            # apply distortion
            x_d = distort_points(np.array([[x, y]]), D, K)
            u, v = x_d[0, :]

            # bilinear interpolation
            u1 = math.floor(u)
            v1 = math.floor(v)
            if bilinear_interpolation:
                a = u - u1
                b = v - v1
                if (u1 >= 0) & (u1+1 < width) & (v1 >= 0) & (v1+1 < height):
                    undistorted_img[y, x] = (1 - b) * (
                        (1 - a) * img[v1, u1] + a * img[v1, u1+1]
                    ) + b * (
                        (1 - a) * img[v1 + 1, u1] + a * img[v1 + 1, u1 + 1]
                        )
            else:
                if (u1 >= 0) & (u1 < width) & (v1 >= 0) & (v1 < height):
                    undistorted_img[y, x] = img[v1, u1]

    return undistorted_img
