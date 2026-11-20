"""Read-only Flexiv RDK pose. One states() call per capture; no enable, mode change, or motion."""
import time
import numpy as np
from .errors import HandeyeError
from .models import PoseReading
from .transforms import from_pose


def tcp_pose_to_base_transform(tcp_pose):
    """Flexiv tcp_pose is TCP in the robot base/world frame: [x, y, z, qw, qx, qy, qz], meters."""
    pose = np.asarray(tcp_pose, dtype=float).reshape(-1)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise HandeyeError("ROBOT_QUERY_FAILED", "Flexiv tcp_pose 必须是 7 个有限数 [x, y, z, qw, qx, qy, qz]")
    x, y, z, qw, qx, qy, qz = pose
    return from_pose([x, y, z], [qx, qy, qz, qw])


def reading_from_tcp_pose(tcp_pose, config, request_id, started_wall_ns, roundtrip_ms):
    pose = np.asarray(tcp_pose, dtype=float).reshape(7)
    base, end = config["frames"]["base"], config["frames"]["end"]
    return PoseReading(tcp_pose_to_base_transform(pose), {
        "request_id": request_id,
        "header": {"frame_id": base, "child_frame_id": end, "stamp": None, "input_inverted": False,
                   "translation_unit": "m", "quaternion_order": "xyzw"},
        "base_frame": base, "end_frame": end,
        "tcp_pose_in_base_wxyz": pose.tolist(),
        "timestamp_kind": "host_query", "source_kind": "flexiv_rdk",
        "source_age_known": False, "has_motion_state": False,
        "query_started_host_wall_ns": started_wall_ns,
        "response_received_host_wall_ns": time.time_ns(),
        "roundtrip_ms": roundtrip_ms,
        "sdk_call_count": 1, "sdk_count_unavailable_reason": None,
    })


class FlexivPoseSource:
    def __init__(self, config):
        serial = str(config["robot"].get("serial", "")).strip()
        if not serial:
            raise HandeyeError("CONFIG", "Flexiv 位姿源需要 robot.serial")
        import flexivrdk
        self.config = config
        self.serial = serial
        self.robot = flexivrdk.Robot(serial)
        if not self.robot.connected():
            self.robot = None
            raise HandeyeError("ROBOT_UNAVAILABLE", f"未连上 Flexiv {serial}")

    def close(self):
        self.robot = None

    def read(self, request_id, timeout, cancel, on_request):
        if self.robot is None:
            raise HandeyeError("ROBOT_UNAVAILABLE", "Flexiv 连接已关闭")
        if cancel.is_set():
            raise HandeyeError("CANCELLED", "已取消")
        deadline = time.monotonic() + timeout
        on_request()
        started_wall_ns = time.time_ns()
        started = time.monotonic()
        states = self.robot.states()
        roundtrip_ms = (time.monotonic() - started) * 1e3
        if cancel.is_set():
            raise HandeyeError("CANCELLED", "已取消")
        if time.monotonic() >= deadline:
            raise HandeyeError("ROBOT_TIMEOUT", "Flexiv 位姿读取超时；未自动重试")
        return reading_from_tcp_pose(states.tcp_pose, self.config, request_id, started_wall_ns, roundtrip_ms)
