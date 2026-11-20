from copy import deepcopy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from handeye_core.config import load_config
from handeye_core.detection import from_corners, project, detect
from handeye_core.errors import HandeyeError
from handeye_core.models import Board
from handeye_core.solver import solve
from handeye_core.synthetic import scene, pose_at, render
from handeye_core.transforms import inverse, transform, rotation_error_deg


def measurements(train=20, validation=5, noise=0):
    board = Board()
    X, Y, camera = scene()
    rng = np.random.default_rng(8912)
    samples = []
    for i in range(train + validation):
        C = pose_at(i, board)
        G = Y @ inverse(C) @ inverse(X)
        pixels = project(board.object_points(), C, camera)
        pixels += rng.normal(0, noise, pixels.shape)
        analysis = from_corners(pixels, board, camera)
        samples.append({"sample_id": str(i), "role": "train" if i < train else "validation",
                        "enabled": True, "G": G.tolist(), "analysis": analysis})
    return samples, X, Y, camera


@pytest.mark.parametrize("method", ["PARK", "TSAI"])
def test_known_extrinsic_and_independent_predictions(method):
    samples, X, Y, camera = measurements()
    result = solve(samples, camera, method)
    predicted = np.asarray(result["T_parent_child"])
    assert np.linalg.norm(predicted[:3, 3] - X[:3, 3]) * 1000 < 0.01
    assert rotation_error_deg(predicted[:3, :3], X[:3, :3]) < 0.001
    assert result["metrics"]["validation"]["predicted_reprojection_px"]["rmse"] < 0.001


def test_validation_does_not_refit_either_transform():
    samples, _, _, camera = measurements()
    baseline = solve(samples, camera)
    corrupted = deepcopy(samples)
    for sample in corrupted[20:]:
        sample["G"][0][3] += 0.04
    result = solve(corrupted, camera)
    np.testing.assert_array_equal(result["T_parent_child"], baseline["T_parent_child"])
    np.testing.assert_array_equal(result["T_base_board_training_mean"], baseline["T_base_board_training_mean"])
    assert result["metrics"]["validation"]["position_mm"]["rmse"] == pytest.approx(40, abs=0.001)
    assert result["metrics"]["validation"]["predicted_reprojection_px"]["rmse"] > 10


def test_noise_and_inverted_robot_pose_are_visible_in_report():
    samples, X, _, camera = measurements(noise=0.12)
    result = solve(samples, camera)
    assert np.linalg.norm(np.array(result["translation"]) - X[:3, 3]) * 1000 < 5
    assert result["metrics"]["validation"]["position_mm"]["rmse"] < 5
    inverted = deepcopy(samples)
    for sample in inverted:
        sample["G"] = inverse(sample["G"]).tolist()
    try:
        bad = solve(inverted, camera)
    except HandeyeError as exc:
        assert exc.code == "SOLVE_FAILED"
    else:
        assert bad["metrics"]["training"]["position_mm"]["rmse"] > 2
        assert bad["warnings"]


def test_two_independent_rotation_directions_are_sufficient():
    board = Board()
    X, Y, camera = scene()
    samples = []
    # Two nonparallel axes are sufficient; a rank-three-only check would reject this.
    for i, vector in enumerate(([0, 0, 0], [0.35, 0, 0], [0, 0.4, 0])):
        G = transform(Rotation.from_rotvec(vector).as_matrix(), [0.1 * i, 0, 0])
        C = inverse(X) @ inverse(G) @ Y
        samples.append({"sample_id": str(i), "role": "train", "G": G.tolist(),
            "analysis": {"C": C.tolist(), "object_points": board.object_points().tolist(),
                         "corners": project(board.object_points(), C, camera).tolist()}})
    result = solve(samples, camera)
    np.testing.assert_allclose(result["T_parent_child"], X, atol=1e-7)
    assert result["metrics"]["validation"]["position_mm"] is None
    assert result["validation_state"] == "computed"


@pytest.mark.parametrize("mode", ["translation", "single_axis", "repeated"])
def test_underdetermined_motion_is_not_a_calibration(mode):
    samples, _, _, camera = measurements()
    for i, sample in enumerate(samples):
        rot = Rotation.from_rotvec([0, 0, i * 0.03]).as_matrix() if mode == "single_axis" else np.eye(3)
        sample["G"] = transform(rot, [i * 0.01, 0, 0] if mode != "repeated" else [0, 0, 0]).tolist()
    with pytest.raises(HandeyeError, match="退化"):
        solve(samples, camera)


def test_board_geometry_and_manual_orientation_correction():
    board, squares = Board(), Board(count_interpretation="squares")
    assert board.object_points().shape == (54, 3)
    assert squares.object_points().shape == (40, 3)
    np.testing.assert_allclose(board.object_points().max(axis=0), [0.344, 0.215, 0])
    _, _, camera = scene()
    C = pose_at(4, board)
    pixels = project(board.object_points(), C, camera)
    flipped = from_corners(pixels[::-1], board, camera, flip=True)
    np.testing.assert_allclose(flipped["C"], C, atol=1e-6)
    image = render(board, camera, C)
    analysis = detect(image, board, camera)
    assert analysis["corner_count"] == 54
    assert analysis["pnp_rmse_px"] < 1
    with pytest.raises(HandeyeError, match="完整"):
        detect(image[:100].copy(), board,
               type(camera)(camera.width, 100, camera.K, camera.D))

