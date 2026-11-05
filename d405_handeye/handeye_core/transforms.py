import numpy as np
from scipy.spatial.transform import Rotation
from .errors import HandeyeError


def checked(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise HandeyeError("INVALID_TRANSFORM", "变换必须是有限的 4×4 矩阵")
    R = T[:3, :3]
    if (not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-5)):
        raise HandeyeError("INVALID_TRANSFORM", "旋转矩阵或齐次矩阵末行无效")
    return T


def transform(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t).reshape(3)
    return checked(T)


def from_pose(translation, quaternion_xyzw):
    q = np.asarray(quaternion_xyzw, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise HandeyeError("INVALID_QUATERNION", "四元数必须有限且非零，顺序为 xyzw")
    return transform(Rotation.from_quat(q / np.linalg.norm(q)).as_matrix(), translation)


def inverse(T):
    T = checked(T)
    R = T[:3, :3].T
    return transform(R, -R @ T[:3, 3])


def rotation_error_deg(A, B):
    return float(np.degrees(Rotation.from_matrix(A.T @ B).magnitude()))


def mean_transform(poses):
    poses = np.asarray([checked(T) for T in poses])
    return transform(Rotation.from_matrix(poses[:, :3, :3]).mean().as_matrix(),
                     poses[:, :3, 3].mean(axis=0))

