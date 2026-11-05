import numpy as np
from handeye_core.models import Board, CameraModel
from handeye_core.camera_geometry import project_camera_points, normalized_pixels
from handeye_core.detection import project, from_corners
from handeye_core.synthetic import pose_at


def inverse_camera():
    return CameraModel(848, 480, [[620, 0, 424], [0, 615, 240], [0, 0, 1]],
                       [0.08, -0.025, 0.002, -0.003, 0.004], "inverse_brown_conrady")


def test_d405_inverse_model_recovers_pose_and_original_pixel_residual():
    camera, board = inverse_camera(), Board()
    C = pose_at(9, board)
    pixels = project(board.object_points(), C, camera)
    analysis = from_corners(pixels, board, camera)
    np.testing.assert_allclose(analysis["C"], C, atol=1e-7)
    assert analysis["pnp_rmse_px"] < 1e-5
    assert analysis["pnp_coordinate_space"] == "sdk_normalized_rays_no_further_distortion"


def test_projection_and_normalized_rays_agree():
    camera = inverse_camera()
    rays = np.array([[-0.5, -0.3, 1], [-0.2, 0.2, 1], [0, 0, 1], [0.5, 0.3, 1]])
    pixels = project_camera_points(rays, camera)
    np.testing.assert_allclose(normalized_pixels(pixels, camera), rays[:, :2], atol=1e-8)


def test_adapter_against_realsense_sdk_without_hardware():
    import pytest
    rs = pytest.importorskip("pyrealsense2")
    camera = inverse_camera()
    intrinsics = rs.intrinsics()
    intrinsics.width, intrinsics.height = camera.width, camera.height
    intrinsics.fx, intrinsics.fy = camera.K[0][0], camera.K[1][1]
    intrinsics.ppx, intrinsics.ppy = camera.K[0][2], camera.K[1][2]
    intrinsics.model, intrinsics.coeffs = rs.distortion.inverse_brown_conrady, camera.D
    rays = np.array([[-0.5, -0.3, 1], [-0.2, 0.2, 1], [0, 0, 1], [0.5, 0.3, 1]])
    expected_pixels = np.array([rs.rs2_project_point_to_pixel(intrinsics, ray.tolist()) for ray in rays])
    pixels = project_camera_points(rays, camera)
    np.testing.assert_allclose(pixels, expected_pixels, atol=1e-4, rtol=0)
    expected_rays = np.array([rs.rs2_deproject_pixel_to_point(intrinsics, pixel.tolist(), 1)[:2]
                             for pixel in pixels])
    np.testing.assert_allclose(normalized_pixels(pixels, camera), expected_rays, atol=1e-6, rtol=0)

