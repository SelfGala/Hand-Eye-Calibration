import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from handeye_core.models import PoseReading
from handeye_core.transforms import from_pose
from .node import RobotPoseService


class MockReader:
    def __init__(self, node):
        self.node = node

    def read_pose(self, timeout_seconds):
        t = list(self.node.get_parameter("translation_m").value)
        q = list(self.node.get_parameter("quaternion_xyzw").value)
        return PoseReading(from_pose(t, q), {"base_frame": "base_link", "end_frame": "l_end_link",
            "timestamp_kind": "host_query_midpoint", "source_kind": "replay",
            "source_age_known": False, "has_motion_state": False, "sdk_call_count": 0})


def main():
    rclpy.init()
    node = Node("mock_handeye_robot_bridge")
    node.declare_parameter("translation_m", [0.0, 0.0, 0.0])
    node.declare_parameter("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])
    node.get_logger().warning("MOCK ONLY：模拟位姿，不连接机械臂，不可用于实际标定")
    bridge = RobotPoseService(node, MockReader(node), {})
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        bridge.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

