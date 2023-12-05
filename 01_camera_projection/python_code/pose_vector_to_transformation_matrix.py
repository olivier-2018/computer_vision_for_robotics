import numpy as np


def pose_vector_to_transformation_matrix(pose_vec: np.ndarray) -> np.ndarray:
    """
    Converts a 6x1 pose vector into a 4x4 transformation matrix.

    Args:
        pose_vec: 6x1 vector representing the pose as [wx, wy, wz, tx, ty, tz]

    Returns:
        T: 4x4 transformation matrix
    """
    pass

    # 3d rotation
    omega = pose_vec[:3]

    # 3d translation
    t = pose_vec[3:]

    # unit vector indicating the axis of rotation
    theta = np.sqrt((omega**2).sum())
    k = omega / theta

    kx, ky, kz = k

    # cross-product matrix
    K = np.array([[0, -kz, ky],
                  [kz, 0, -kx],
                  [-ky, kx, 0]])

    # Rodrigues' rotation formula
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T
