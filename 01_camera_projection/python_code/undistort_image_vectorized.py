import numpy as np

from distort_points import distort_points


def undistort_image_vectorized(img: np.ndarray,
                               K: np.ndarray,
                               D: np.ndarray) -> np.ndarray:

    """
    Undistorts an image using the camera matrix and distortion coefficients.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        und_img: undistorted image (HxW)
    """
    pass

    height, width = img.shape[-2:]
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    px_locs = np.stack([X, Y], axis=-1).reshape([height * width, 2])

    dist_px_locs = distort_points(px_locs, D, K)
    intensity_vals = img[np.round(dist_px_locs[:, 1].astype(int)),
                         np.round(dist_px_locs[:, 0]).astype(int)]
    und_img = intensity_vals.reshape(img.shape).astype(np.uint8)

    return und_img
