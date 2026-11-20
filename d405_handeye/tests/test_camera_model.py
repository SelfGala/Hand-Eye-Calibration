import numpy as np
from handeye_core.models import Board, CameraModel
from handeye_core.camera_geometry import project_camera_points, normalized_pixels
from handeye_core.detection import project, from_corners
from handeye_core.synthetic import pose_at


def inverse_camera():
    return CameraModel(848, 480, [[620, 0, 424], [0, 615, 240], [0, 0, 1]],
                       [0.08, -0.025, 0.002, -0.003, 0.004], "inverse_brown_conrady")


def test_configured_intrinsics_are_used_as_opencv_model():
    from handeye_core.camera import camera_model_from_config
    from handeye_core.errors import HandeyeError
    cfg = {"width": 1280, "height": 720, "distortion_model": "opencv",
           "K": [657.2001944551392, 0.0, 639.2417587535549,
                 0.0, 657.0404307214897, 364.38314456683963, 0.0, 0.0, 1.0],
           "D": [-0.05242405416347785, 0.06063636028297523, -0.0006438179867858043,
                 0.001848309771830554, -0.02075842916066004]}
    model = camera_model_from_config(cfg, {"intrinsics_source": "config"})
    assert model.distortion_model == "opencv"
    assert model.width == 1280 and model.height == 720
    np.testing.assert_allclose(model.K[0][0], 657.2001944551392)
    np.testing.assert_allclose(model.D[0], -0.05242405416347785)
    try:
        camera_model_from_config({"width": 1280, "height": 720}, {})
    except HandeyeError as exc:
        assert exc.code == "CONFIG"
    else:
        raise AssertionError("显式配置路径缺少内参时应当失败")


def test_device_intrinsics_are_copied_as_reported():
    from types import SimpleNamespace
    from handeye_core.camera import camera_model_from_realsense
    from handeye_core.errors import HandeyeError
    intrinsics = SimpleNamespace(width=1280, height=720, fx=600.5, fy=601.25, ppx=640.0, ppy=360.0,
                                 model="distortion.inverse_brown_conrady",
                                 coeffs=[0.1, -0.02, 0.001, -0.002, 0.003, 9.0])
    model = camera_model_from_realsense(intrinsics, {"serial": "034522060350"})
    assert model.distortion_model == "inverse_brown_conrady"
    assert model.profile["intrinsics_source"] == "device"
    np.testing.assert_allclose(model.K[0][0], 600.5)
    np.testing.assert_allclose(model.D, [0.1, -0.02, 0.001, -0.002, 0.003])
    intrinsics.model = "kannala_brandt4"
    try:
        camera_model_from_realsense(intrinsics, {})
    except HandeyeError as exc:
        assert exc.code == "CAMERA_MODEL_UNSUPPORTED"
    else:
        raise AssertionError("未适配的相机畸变模型不应当被改写成别的模型")


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

