from pathlib import Path
from threading import Event
from types import SimpleNamespace
import numpy as np
from handeye_core.config import load_config
from handeye_core.flexiv_pose import FlexivPoseSource, tcp_pose_to_base_transform
from handeye_core.transforms import pose_components


def test_tcp_pose_is_base_to_tcp_with_xyzw():
    transform = tcp_pose_to_base_transform([0.1, -0.2, 0.3, 0.5, 0.5, 0.5, 0.5])
    pose = pose_components(transform)
    np.testing.assert_allclose(pose["translation_m"], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(pose["quaternion_xyzw"], [0.5, 0.5, 0.5, 0.5])


def test_flexiv_config_omits_intrinsics_and_reads_tcp_once():
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "d405_flexiv_eye_in_hand.yaml")
    assert config["camera"]["serial"] == "034522060350"
    assert "K" not in config["camera"] and "D" not in config["camera"]
    assert config["robot"]["pose_source"] == "flexiv_sdk"
    assert config["robot"]["serial"] == "Rizon4s-063191"
    assert config["frames"]["base"] == "flexiv_base"
    assert config["frames"]["end"] == "flexiv_tcp"

    class Robot:
        def __init__(self):
            self.calls = 0

        def states(self):
            self.calls += 1
            return SimpleNamespace(tcp_pose=[0.4, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])

    robot = Robot()
    source = FlexivPoseSource.__new__(FlexivPoseSource)
    source.config = config
    source.serial = config["robot"]["serial"]
    source.robot = robot
    calls = []
    reading = source.read("pose-1", 1.5, Event(), lambda: calls.append(1))
    assert robot.calls == 1 and calls == [1]
    assert reading.metadata["source_kind"] == "flexiv_rdk"
    assert reading.metadata["sdk_call_count"] == 1
    assert reading.metadata["header"]["frame_id"] == "flexiv_base"
    assert reading.metadata["header"]["child_frame_id"] == "flexiv_tcp"
    np.testing.assert_allclose(reading.metadata["tcp_pose_in_base_wxyz"], [0.4, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(reading.G[:3, 3], [0.4, 0.0, 0.2])
    np.testing.assert_allclose(reading.G[:3, :3], np.eye(3))
