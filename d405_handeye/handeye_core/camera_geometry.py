"""Pixel geometry, including the D405 inverse_brown_conrady SDK convention.

The RealSense-specific branch follows rs2_project_point_to_pixel and
rs2_deproject_pixel_to_point, rather than treating its coefficients as OpenCV D.
Reference and model provenance: docs/camera_model.md.
"""
import cv2
import numpy as np
from .errors import HandeyeError


def normalized_pixels(pixels, camera):
    pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
    K = np.asarray(camera.K)
    if camera.distortion_model != "inverse_brown_conrady":
        return cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, np.asarray(camera.D)).reshape(-1, 2)
    xy0 = (pixels - [K[0, 2], K[1, 2]]) / [K[0, 0], K[1, 1]]
    xy = xy0.copy()
    k1, k2, p1, p2, k3 = camera.D
    # Fixed-point inversion of the SDK's radial-then-tangential forward map.
    for _ in range(10):
        x, y = xy.T
        radius2 = x * x + y * y
        radial = 1 + radius2 * (k1 + radius2 * (k2 + radius2 * k3))
        x_radial, y_radial = x * radial, y * radial
        tangent = np.column_stack((2 * p1 * x_radial * y_radial + p2 * (radius2 + 2 * x_radial**2),
                                   2 * p2 * x_radial * y_radial + p1 * (radius2 + 2 * y_radial**2)))
        xy = (xy0 - tangent) / radial[:, None]
    if not np.isfinite(xy).all():
        raise HandeyeError("CAMERA_MODEL", "反向 Brown-Conrady 归一化未得到有效坐标")
    return xy


def project_camera_points(points, camera):
    points = np.asarray(points, np.float64).reshape(-1, 3)
    K = np.asarray(camera.K)
    if camera.distortion_model != "inverse_brown_conrady":
        return cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, np.asarray(camera.D))[0].reshape(-1, 2)
    x, y = (points[:, :2] / points[:, 2:3]).T
    k1, k2, p1, p2, k3 = camera.D
    radius2 = x * x + y * y
    radial = 1 + radius2 * (k1 + radius2 * (k2 + radius2 * k3))
    xr, yr = x * radial, y * radial
    distorted = np.column_stack((xr + 2 * p1 * xr * yr + p2 * (radius2 + 2 * xr**2),
                                 yr + 2 * p2 * xr * yr + p1 * (radius2 + 2 * yr**2)))
    return distorted * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]

