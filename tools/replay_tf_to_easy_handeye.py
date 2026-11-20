#!/usr/bin/python3.10
"""把 d405_handeye 会话样本逐条发到 /tf，配合 easy_handeye2 的 Sample 按钮。

每条样本持续发布两段变换，时间戳用当前时间：
  base_link    -> r_end_link    来自 sample.json 的 G
  camera_link  -> plant_link    来自 sample.json 的 analysis.C

easy_handeye2 每成功收下一条样本，这里就切到下一条姿态，直到修订索引里的样本发完。
删除最新一条样本时会退回对应姿态。一次只点一下 Sample。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rclpy
from easy_handeye2_msgs.srv import TakeSample
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

DEFAULT_SESSION = Path(__file__).resolve().parents[1] / "examples" / "synthetic_session"


def quat_xyzw(rotation):
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([x, y, z, w], dtype=float)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("旋转矩阵无法转成四元数")
    return quaternion / norm


def load_session(session_dir):
    revisions = sorted((session_dir / "revisions").glob("index_*.json"))
    if not revisions:
        raise FileNotFoundError(f"找不到修订索引: {session_dir / 'revisions'}")
    index_path = revisions[-1]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    poses = []
    for entry in index["samples"]:
        if not entry.get("enabled", True):
            continue
        path = session_dir / "samples" / entry["sample_id"] / "sample.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        robot = np.asarray(data["G"], dtype=float)
        marker = np.asarray(data["analysis"]["C"], dtype=float)
        if robot.shape != (4, 4) or marker.shape != (4, 4):
            raise ValueError(f"{path} 的 G 或 analysis.C 不是 4x4")
        poses.append({
            "label": entry.get("label") or data.get("label") or entry["sample_id"],
            "role": entry.get("role") or data.get("role") or "",
            "sample_id": entry["sample_id"],
            "robot_t": robot[:3, 3].copy(),
            "robot_q": quat_xyzw(robot[:3, :3]),
            "marker_t": marker[:3, 3].copy(),
            "marker_q": quat_xyzw(marker[:3, :3]),
        })
    if not poses:
        raise RuntimeError(f"{session_dir} 里没有可发布的样本")
    return index_path.name, poses


class ReplayNode(Node):
    def __init__(self, poses, args):
        super().__init__("replay_tf_to_easy_handeye")
        self.poses = poses
        self.base = args.base_frame
        self.end = args.end_frame
        self.camera = args.camera_frame
        self.marker = args.marker_frame
        self.service_name = args.sample_list_service
        self.index = 0
        self.last_count = None
        self.pending = None
        self.finished_logged = False
        self.service_warned = False

        self.broadcaster = TransformBroadcaster(self)
        self.client = self.create_client(TakeSample, self.service_name)
        self.create_timer(1.0 / args.rate, self._publish)
        self.create_timer(0.05, self._poll)
        self._announce("开始发布")

    def _fill(self, parent, child, translation, quaternion, stamp):
        message = TransformStamped()
        message.header.stamp = stamp
        message.header.frame_id = parent
        message.child_frame_id = child
        message.transform.translation.x = float(translation[0])
        message.transform.translation.y = float(translation[1])
        message.transform.translation.z = float(translation[2])
        message.transform.rotation.x = float(quaternion[0])
        message.transform.rotation.y = float(quaternion[1])
        message.transform.rotation.z = float(quaternion[2])
        message.transform.rotation.w = float(quaternion[3])
        return message

    def _publish(self):
        pose = self.poses[self.index]
        stamp = self.get_clock().now().to_msg()
        self.broadcaster.sendTransform([
            self._fill(self.base, self.end, pose["robot_t"], pose["robot_q"], stamp),
            self._fill(self.camera, self.marker, pose["marker_t"], pose["marker_q"], stamp),
        ])

    def _announce(self, reason):
        pose = self.poses[self.index]
        robot = pose["robot_t"]
        marker = pose["marker_t"]
        self.get_logger().info(
            f"{reason}: [{self.index + 1}/{len(self.poses)}] {pose['label']} ({pose['role']})  "
            f"{self.base}->{self.end} "
            f"t=({robot[0]:.4f}, {robot[1]:.4f}, {robot[2]:.4f})  "
            f"{self.camera}->{self.marker} "
            f"t=({marker[0]:.4f}, {marker[1]:.4f}, {marker[2]:.4f})"
        )

    def _seek(self, count):
        if count >= len(self.poses):
            target = len(self.poses) - 1
            if self.index != target:
                self.index = target
                self._announce("保持最后一条")
            if not self.finished_logged:
                self.finished_logged = True
                self.get_logger().info(
                    f"{len(self.poses)} 条都已采进 easy_handeye2，可以点 Compute。"
                    "本脚本继续发最后一条 TF。"
                )
            return
        self.finished_logged = False
        if self.index != count:
            self.index = count
            self._announce("切换")

    def _poll(self):
        if not self.client.service_is_ready():
            if not self.service_warned:
                self.service_warned = True
                self.get_logger().warn(
                    f"还没看到 {self.service_name}，先按第 1 条发 TF。请先启动 easy_handeye2。"
                )
            return
        if self.pending is not None:
            if not self.pending.done():
                return
            future = self.pending
            self.pending = None
            if future.exception() is not None:
                self.get_logger().error(f"读取样本列表失败: {future.exception()}")
                return
            count = len(future.result().samples.samples)
            previous = self.last_count
            self.last_count = count
            if previous is None:
                self._seek(count)
                if count == 0:
                    self.get_logger().info("easy_handeye2 当前 0 条样本。点 Sample 后会自动切到下一条。")
                elif count < len(self.poses):
                    self.get_logger().info(f"easy_handeye2 已有 {count} 条，从第 {count + 1} 条继续。")
                return
            if count == previous:
                return
            if count > previous + 1:
                self.get_logger().warn(
                    f"样本数一次从 {previous} 跳到 {count}。"
                    "多出来的几条用的是同一次 TF，建议删掉后一次只点一下 Sample。"
                )
            else:
                self.get_logger().info(f"样本数 {previous} -> {count}")
            self._seek(count)
            return
        self.pending = self.client.call_async(TakeSample.Request())


def main():
    parser = argparse.ArgumentParser(description="按 easy_handeye2 的 Sample 逐条重放 d405_handeye 样本 TF")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION, help="含 revisions/ 和 samples/ 的会话目录")
    parser.add_argument("--rate", type=float, default=100.0, help="TF 发布频率，默认 100 Hz")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--end-frame", default="r_end_link")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--marker-frame", default="plant_link")
    parser.add_argument("--sample-list-service", default="/easy_handeye2/calibration/get_sample_list")
    args = parser.parse_args()
    if args.rate <= 0:
        parser.error("--rate 必须大于 0")

    revision, poses = load_session(args.session.resolve())
    print(f"会话: {args.session.resolve()}")
    print(f"修订: {revision}，启用样本 {len(poses)} 条")
    for pose in poses:
        print(f"  {pose['label']:16} {pose['role']:10} {pose['sample_id']}")

    rclpy.init()
    node = ReplayNode(poses, args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
