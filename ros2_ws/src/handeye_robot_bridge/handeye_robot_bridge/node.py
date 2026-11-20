from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Lock
import importlib
import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from handeye_interfaces.srv import GetEndPose
from std_srvs.srv import Trigger
from handeye_core.errors import HandeyeError
from handeye_core.transforms import checked
from handeye_core.ros_runtime import fill_transform


class RobotPoseService:
    """reader.read_pose(timeout_seconds) performs exactly one bounded driver access.

    Reuse the driver's own lock/connection inside the reader. This extra lock prevents
    overlapping requests from this bridge, including a timed-out SDK call still running.
    """
    def __init__(self, node, reader, config):
        self.node, self.reader, self.config = node, reader, config
        self.lock = Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robot-on-request")
        self.request_count = 0
        self.adapter_read_count = 0
        self.sdk_call_count = 0
        self.sdk_count_complete = True
        self.group = ReentrantCallbackGroup()
        self.service = node.create_service(GetEndPose, config.get("service_name", "/robot/get_end_pose"),
                                           self.query, callback_group=self.group)
        self.counter_service = node.create_service(Trigger, "~/query_counts", self.counts, callback_group=self.group)

    def _read(self, timeout):
        try:
            self.adapter_read_count += 1
            value = self.reader.read_pose(timeout_seconds=timeout)
            count = value.metadata.get("sdk_call_count")
            if count is None:
                self.sdk_count_complete = False
            else:
                self.sdk_call_count += int(count)
            return value
        finally:
            self.lock.release()

    def query(self, request, response):
        response.request_id = request.request_id
        self.request_count += 1
        if (request.base_frame, request.end_frame) != (self.config.get("base_frame", "base_link"),
                                                      self.config.get("end_frame", "l_end_link")):
            response.error_code, response.message = "FRAME_MISMATCH", "请求坐标系与适配器不一致"
            return response
        if not self.lock.acquire(blocking=False):
            response.error_code, response.message = "ROBOT_BUSY", "前次查询或驱动仍忙"
            return response
        timeout = float(self.config.get("sdk_timeout_seconds", 0.8))
        if timeout <= 0:
            self.lock.release()
            response.error_code, response.message = "CONFIG", "SDK 超时必须大于零"
            return response
        response.query_started_at = self.node.get_clock().now().to_msg()
        future = self.pool.submit(self._read, timeout)
        try:
            reading = future.result(timeout=timeout)
            T = checked(reading.G)
            metadata = reading.metadata
            if metadata.get("base_frame") != request.base_frame or metadata.get("end_frame") != request.end_frame:
                raise HandeyeError("FRAME_MISMATCH", "reader 必须声明真实坐标系，禁止仅重命名 TCP")
            response.timestamp_kind = metadata["timestamp_kind"]
            response.source_kind = metadata["source_kind"]
            response.source_age_known = bool(metadata.get("source_age_known", False))
            age = metadata.get("source_age_ms")
            response.source_age_ms = float(age) if response.source_age_known else 0.0
            response.has_motion_state = bool(metadata.get("has_motion_state", False))
            response.is_moving = bool(metadata.get("is_moving", False))
            sdk_count = metadata.get("sdk_call_count")
            response.sdk_call_count_known = sdk_count is not None
            response.sdk_call_count = int(sdk_count or 0)
            finished = self.node.get_clock().now().to_msg()
            response.query_finished_at = finished
            stamp = metadata.get("sample_stamp")
            if stamp is None:
                stamp = self.node.get_clock().now().to_msg()
                start_ns = response.query_started_at.sec * 10**9 + response.query_started_at.nanosec
                finish_ns = finished.sec * 10**9 + finished.nanosec
                stamp.sec, stamp.nanosec = divmod((start_ns + finish_ns) // 2, 10**9)
                if response.timestamp_kind == "sdk_sample_time":
                    raise HandeyeError("TIMESTAMP", "sdk_sample_time 必须提供真实 sample_stamp")
            fill_transform(response.pose, T, request.base_frame, request.end_frame, stamp)
            response.success = True
        except TimeoutError:
            response.error_code, response.message = "ROBOT_TIMEOUT", "SDK 查询超时，未重试；锁保留至 SDK 返回"
        except Exception as exc:
            response.error_code, response.message = getattr(exc, "code", "ROBOT_QUERY_FAILED"), str(exc)
        finally:
            if not response.query_finished_at.sec and not response.query_finished_at.nanosec:
                response.query_finished_at = self.node.get_clock().now().to_msg()
        return response

    def counts(self, request, response):
        response.success = True
        response.message = json.dumps({"business_requests": self.request_count,
            "adapter_reads": self.adapter_read_count, "sdk_call_count": self.sdk_call_count,
            "sdk_count_complete": self.sdk_count_complete, "busy": self.lock.locked()})
        return response

    def close(self):
        self.node.destroy_service(self.service)
        self.node.destroy_service(self.counter_service)
        self.pool.shutdown(wait=False)


def main():
    import yaml
    rclpy.init()
    node = Node("handeye_robot_bridge")
    bridge = executor = None
    try:
        path = node.declare_parameter("config_path", "").value
        if not path:
            raise RuntimeError("必须提供 config_path；没有真实 reader 时请显式运行 mock_pose_service")
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        module, symbol = config["reader_factory"].split(":", 1)
        factory = getattr(importlib.import_module(module), symbol)
        reader = factory(config.get("reader_options", {}))
        bridge = RobotPoseService(node, reader, config)
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if executor:
            executor.shutdown()
        if bridge:
            bridge.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

