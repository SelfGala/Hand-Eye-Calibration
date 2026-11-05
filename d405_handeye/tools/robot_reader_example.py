"""Copy into the existing driver project and bind to its measured-state read API."""
from handeye_core.models import PoseReading
from handeye_core.transforms import from_pose
from handeye_core.errors import HandeyeError


class ExistingDriverReader:
    def __init__(self, measured_pose_function):
        # This function owns the actual driver connection and its original lock.
        self.measured_pose_function = measured_pose_function

    def read_pose(self, timeout_seconds):
        # Exactly one call. The driver must implement a real finite SDK timeout.
        state = self.measured_pose_function(timeout_seconds=timeout_seconds)
        # Example input contract only; replace with the real SDK's measured pose.
        if state["base_frame"] != "base_link" or state["end_frame"] != "r_end_link":
            raise HandeyeError("FRAME_MISMATCH", "请先按已知外参转换实际 TCP")
        T = from_pose(state["translation_m"], state["quaternion_xyzw"])
        return PoseReading(T, {"base_frame": "base_link", "end_frame": "r_end_link",
            "timestamp_kind": "host_query_midpoint", "source_kind": "sdk_on_request",
            "source_age_known": False, "has_motion_state": False,
            "sdk_call_count": state.get("sdk_call_count")})


def attach_to_existing_node(node, measured_pose_function):
    from handeye_robot_bridge.node import RobotPoseService
    return RobotPoseService(node, ExistingDriverReader(measured_pose_function), {
        "service_name": "/robot/get_end_pose", "base_frame": "base_link", "end_frame": "r_end_link",
        "sdk_timeout_seconds": 0.8})

