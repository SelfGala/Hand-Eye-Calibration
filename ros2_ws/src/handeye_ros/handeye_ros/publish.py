import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from handeye_core.transforms import checked, inverse
from handeye_core.ros_runtime import fill_transform


def main():
    rclpy.init()
    node = Node("publish_handeye_calibration")
    try:
        result_path = node.declare_parameter("result_path", "").value
        link = node.declare_parameter("camera_link_frame", "").value
        extra_path = node.declare_parameter("camera_link_to_optical_path", "").value
        with open(result_path, encoding="utf-8") as f:
            result = yaml.safe_load(f)
        if result.get("schema_version") != 2 or result.get("calibration_type") != "eye_in_hand":
            raise ValueError("不支持的标定结果格式")
        if result.get("camera", {}).get("profile", {}).get("source") == "SYNTHETIC_DEMO":
            raise ValueError("合成演示结果不能用于发布真实标定 TF")
        T = checked(result["T_parent_child"])
        parent, child = result["parent_frame"], result["child_frame"]
        if link:
            if not extra_path:
                raise ValueError("发布 camera_link 需要可信的 T_camera_link_optical 文件")
            with open(extra_path, encoding="utf-8") as f:
                extra = yaml.safe_load(f)
            if extra["parent_frame"] != link or extra["child_frame"] != child:
                raise ValueError("camera_link 与光学帧外参的坐标系不匹配")
            T = T @ inverse(checked(extra["T_parent_child"]))
            child = link
        if parent == child:
            raise ValueError("父子坐标系不能相同")
        message = TransformStamped()
        fill_transform(message, T, parent, child, node.get_clock().now().to_msg())
        broadcaster = StaticTransformBroadcaster(node)
        broadcaster.sendTransform(message)
        node.get_logger().info(f"发布 {parent} <- {child}，结果 {result['result_id']}")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

