"""ROS is imported only on a hardware/ROS entry path. Executor runs independently of Qt."""
from threading import Event, Lock, Thread
import json
import time
import numpy as np
from .errors import HandeyeError
from .models import PoseReading
from .transforms import from_pose, inverse


def _stamp(stamp):
    return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}


def _stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def decode_transform(message, config, pose_stamped=False):
    base, end = config["frames"]["base"], config["frames"]["end"]
    if pose_stamped:
        parent, child = message.header.frame_id, end
        pose = message.pose
        t, q = pose.position, pose.orientation
    else:
        parent, child = message.header.frame_id, message.child_frame_id
        t, q = message.transform.translation, message.transform.rotation
    T = from_pose([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
    inverted = False
    if (parent, child) != (base, end):
        if (not pose_stamped and config["robot"]["allow_inverted_input"] and (parent, child) == (end, base)):
            T = inverse(T)
            inverted = True
        else:
            raise HandeyeError("FRAME_MISMATCH", f"收到 {parent} <- {child}，期望 {base} <- {end}")
    return T, {"frame_id": parent, "child_frame_id": child, "stamp": _stamp(message.header.stamp),
               "input_inverted": inverted, "translation_unit": "m", "quaternion_order": "xyzw"}


class ServicePoseSource:
    def __init__(self, node, config, group):
        from handeye_interfaces.srv import GetEndPose
        self.node, self.config = node, config
        self.service_type = GetEndPose
        self.client = node.create_client(GetEndPose, config["robot"]["service_name"], callback_group=group)

    def service_available(self, timeout=0.0):
        return self.client.wait_for_service(timeout_sec=timeout)

    def read(self, request_id, timeout, cancel, on_request):
        deadline = time.monotonic() + timeout
        while not self.client.service_is_ready():
            if cancel.is_set():
                raise HandeyeError("CANCELLED", "已取消")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HandeyeError("ROBOT_SERVICE_UNAVAILABLE", "位姿服务不可用")
            self.client.wait_for_service(timeout_sec=min(0.05, remaining))
        request = self.service_type.Request()
        request.request_id = request_id
        request.base_frame = self.config["frames"]["base"]
        request.end_frame = self.config["frames"]["end"]
        request.require_fresh_sample = True
        started_wall_ns, started_ns = time.time_ns(), time.monotonic_ns()
        on_request()
        future = self.client.call_async(request)
        done = Event()
        future.add_done_callback(lambda _: done.set())
        try:
            while not done.wait(0.02):
                if cancel.is_set():
                    raise HandeyeError("CANCELLED", "已取消")
                if time.monotonic() >= deadline:
                    raise HandeyeError("ROBOT_TIMEOUT", "位姿查询超时；未自动重试")
            if cancel.is_set():
                raise HandeyeError("CANCELLED", "已取消")
            response = future.result()
            if response.request_id != request_id:
                raise HandeyeError("REQUEST_MISMATCH", "响应 request_id 与本次拍照不一致")
            if not response.success:
                raise HandeyeError(response.error_code or "ROBOT_QUERY_FAILED", response.message)
            T, header = decode_transform(response.pose, self.config)
            metadata = {"request_id": request_id, "header": header,
                        "query_started_at": _stamp(response.query_started_at),
                        "query_finished_at": _stamp(response.query_finished_at),
                        "timestamp_kind": response.timestamp_kind, "source_kind": response.source_kind,
                        "source_age_known": response.source_age_known,
                        "source_age_ms": response.source_age_ms if response.source_age_known else None,
                        "has_motion_state": response.has_motion_state,
                        "is_moving": response.is_moving if response.has_motion_state else None,
                        "request_sent_host_wall_ns": started_wall_ns, "request_sent_host_monotonic_ns": started_ns,
                        "response_received_host_wall_ns": time.time_ns(),
                        "roundtrip_ms": (time.monotonic_ns() - started_ns) / 1e6,
                        "sdk_call_count": response.sdk_call_count if response.sdk_call_count_known else None,
                        "sdk_count_unavailable_reason": None if response.sdk_call_count_known else "server did not report count"}
            return PoseReading(T, metadata)
        finally:
            # Pending request removal prevents unbounded futures; a late result belongs to this future only.
            self.client.remove_pending_request(future)
            if not future.done():
                future.cancel()


class TopicPoseSource:
    def __init__(self, node, config, group):
        self.node, self.config, self.group = node, config, group

    def service_available(self, timeout=0.0):
        return None

    def read(self, request_id, timeout, cancel, on_request):
        from geometry_msgs.msg import PoseStamped, TransformStamped
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        types = {"geometry_msgs/msg/PoseStamped": PoseStamped,
                 "geometry_msgs/msg/TransformStamped": TransformStamped}
        type_name = self.config["robot"]["compat_message_type"]
        if type_name not in types:
            raise HandeyeError("MESSAGE_TYPE", "自定义消息需要显式编写适配器")
        started_ros_ns = self.node.get_clock().now().nanoseconds
        event, guard = Event(), Lock()
        value = {}
        def receive(message):
            with guard:
                if value:
                    return
                # A post-subscription message must carry a comparable, new ROS timestamp.
                age_ns = self.node.get_clock().now().nanoseconds - _stamp_ns(message.header.stamp)
                if _stamp_ns(message.header.stamp) < started_ros_ns or age_ns < 0:
                    return
                if age_ns > self.config["robot"]["max_source_age_seconds"] * 1e9:
                    return
                try:
                    T, header = decode_transform(message, self.config, type_name.endswith("PoseStamped"))
                    value["reading"] = PoseReading(T, {"request_id": request_id, "header": header,
                        "source_kind": "topic_compat", "timestamp_kind": "message_header",
                        "source_age_known": True, "source_age_ms": age_ns / 1e6,
                        "has_motion_state": False, "sdk_call_count": None,
                        "sdk_count_unavailable_reason": "publisher owns driver; cannot infer SDK calls",
                        "received_count": 1})
                except Exception as exc:
                    value["error"] = exc
                event.set()
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        sub = self.node.create_subscription(types[type_name], self.config["robot"]["compat_topic"],
                                             receive, qos, callback_group=self.group)
        # Subscription is not a robot business request; GetJob.pose_request_count remains zero.
        try:
            deadline = time.monotonic() + self.config["robot"]["compat_subscription_timeout_seconds"]
            while not event.wait(0.02):
                if cancel.is_set():
                    raise HandeyeError("CANCELLED", "已取消")
                if time.monotonic() >= deadline:
                    raise HandeyeError("TOPIC_TIMEOUT", "未收到本轮新时间戳的位姿，请核对发布和两机时钟")
            if "error" in value:
                raise value["error"]
            return value["reading"]
        finally:
            self.node.destroy_subscription(sub)


def _persistent_listener(base):
    class PersistentTransformListener(base):
        def __init__(self, buffer, node):
            super().__init__(buffer, node, spin_thread=False)
            self._closed = False

        def unregister(self):
            if self._closed:
                return
            self._closed = True
            super().unregister()

        def __del__(self):
            try:
                self.unregister()
            except Exception:
                pass

    return PersistentTransformListener


class TfPoseSource:
    def __init__(self, node, config, group):
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformListener
        self.node, self.config = node, config
        self._time = Time
        self.buffer = Buffer()
        # Keep /tf subscribed for the node lifetime. Destroying it on each capture races the executor,
        # kills the spin thread, and every later lookup then fails with TF_TIMEOUT.
        self.listener = _persistent_listener(TransformListener)(self.buffer, node)

    def close(self):
        listener = getattr(self, "listener", None)
        self.listener = None
        if listener is not None:
            listener.unregister()

    def service_available(self, timeout=0.0):
        return self.probe(timeout)["available"]

    def probe(self, timeout):
        from tf2_ros import TransformException
        base, end = self.config["frames"]["base"], self.config["frames"]["end"]
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                message = self.buffer.lookup_transform(base, end, self._time())
                return {"available": True, "base": base, "end": end, "stamp": _stamp(message.header.stamp)}
            except TransformException:
                if time.monotonic() >= deadline:
                    return {"available": False, "base": base, "end": end}
                time.sleep(0.05)

    def read(self, request_id, timeout, cancel, on_request):
        from tf2_ros import TransformException
        started_ns = self.node.get_clock().now().nanoseconds
        deadline = time.monotonic() + self.config["robot"]["compat_subscription_timeout_seconds"]
        while True:
            if cancel.is_set():
                raise HandeyeError("CANCELLED", "已取消")
            if time.monotonic() >= deadline:
                raise HandeyeError("TF_TIMEOUT", "未收到本轮 base 到 end 的新动态 TF")
            try:
                message = self.buffer.lookup_transform(self.config["frames"]["base"],
                                                       self.config["frames"]["end"], self._time())
                stamp_ns = _stamp_ns(message.header.stamp)
                age_ns = self.node.get_clock().now().nanoseconds - stamp_ns
                if stamp_ns < started_ns or age_ns < 0 or age_ns > self.config["robot"]["max_source_age_seconds"] * 1e9:
                    cancel.wait(0.02)
                    continue
                T, header = decode_transform(message, self.config)
                return PoseReading(T, {"request_id": request_id, "header": header,
                    "source_kind": "tf_compat", "timestamp_kind": "tf_header",
                    "source_age_known": True, "source_age_ms": age_ns / 1e6,
                    "has_motion_state": False, "sdk_call_count": None,
                    "sdk_count_unavailable_reason": "TF publisher owns driver; cannot infer SDK calls"})
            except TransformException:
                cancel.wait(0.02)


class RosRuntime:
    def __init__(self, config, ros_args=None):
        import rclpy
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.callback_groups import ReentrantCallbackGroup
        self.config = config
        self.context = Context()
        rclpy.init(args=ros_args, context=self.context)
        self.node = Node("handeye_calibrator", context=self.context)
        self.executor = MultiThreadedExecutor(num_threads=2, context=self.context)
        self.executor.add_node(self.node)
        self.group = ReentrantCallbackGroup()
        self.services = []
        mode = config["robot"]["pose_source"]
        classes = {"service_request": ServicePoseSource, "topic_snapshot": TopicPoseSource, "tf_snapshot": TfPoseSource}
        if mode not in classes:
            self.node.destroy_node()
            rclpy.shutdown(context=self.context)
            raise HandeyeError("POSE_SOURCE", f"不支持 {mode}")
        try:
            self.pose_source = classes[mode](self.node, config, self.group)
        except BaseException:
            self.node.destroy_node()
            rclpy.shutdown(context=self.context)
            raise
        self._stop = Event()

        def spin():
            from rclpy.executors import ExternalShutdownException
            import traceback
            while not self._stop.is_set():
                try:
                    self.executor.spin()
                    return
                except ExternalShutdownException:
                    return
                except Exception:
                    if self._stop.is_set():
                        return
                    traceback.print_exc()
                    time.sleep(0.05)
        self.thread = Thread(target=spin, name="ros-executor", daemon=True)
        self.thread.start()

    def bind(self, controller):
        from handeye_interfaces.srv import CaptureSample, StartCompute, GetJob, SetSampleEnabled, GetResult
        from std_srvs.srv import Trigger
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
        def guarded(function):
            def callback(request, response):
                try:
                    function(request, response)
                except Exception as exc:
                    if hasattr(response, "success"):
                        response.success = False
                    if hasattr(response, "accepted"):
                        response.accepted = False
                    if hasattr(response, "error_code"):
                        response.error_code = getattr(exc, "code", "INTERNAL_ERROR")
                    if hasattr(response, "message"):
                        response.message = str(exc)
                return response
            return callback
        def capture(q, r):
            r.job_id = controller.capture(q.role, q.label, q.flip_180)
            r.accepted = True
        def compute(q, r):
            r.job_id = controller.compute(q.methods)
            r.accepted = True
        def get_job(q, r):
            job = controller.get_job(q.job_id)
            r.success, r.job_id, r.state = True, q.job_id, job["state"]
            r.sample_id, r.result_ids = job["sample_id"], job["result_ids"]
            r.error_code, r.message = job["error_code"], job["message"]
            r.pose_request_count = job["pose_request_count"]
        def set_enabled(q, r):
            controller.set_sample(q.sample_id, enabled=q.enabled)
            r.success = True
            r.dataset_revision = controller.dataset.revision
        def get_result(q, r):
            result = controller.get_result(q.result_id)
            r.success = True
            r.result_id, r.method = result["result_id"], result["method"]
            r.dataset_revision = result["dataset_revision"]
            r.stale = result["stale"]
            r.training_count, r.validation_count = len(result["training_sample_ids"]), len(result["validation_sample_ids"])
            r.has_validation_metrics = result["has_validation_metrics"]
            r.training_position_rmse_mm = result["metrics"]["training"]["position_mm"]["rmse"]
            r.training_rotation_rmse_deg = result["metrics"]["training"]["rotation_deg"]["rmse"]
            if result["has_validation_metrics"]:
                r.validation_position_rmse_mm = result["metrics"]["validation"]["position_mm"]["rmse"]
                r.validation_rotation_rmse_deg = result["metrics"]["validation"]["rotation_deg"]["rmse"]
            r.validation_state, r.pairing_mode = result["validation_state"], result["pairing_mode"]
            fill_transform(r.camera_to_end, np.asarray(result["T_parent_child"]), result["parent_frame"], result["child_frame"])
            r.report_json = json.dumps(result, ensure_ascii=False, allow_nan=False)
        def new_session(q, r):
            controller.new_session()
            r.success, r.message = True, str(controller.dataset.path)
        for name, kind, function in (("capture", CaptureSample, capture), ("compute", StartCompute, compute),
                                      ("get_job", GetJob, get_job), ("set_sample_enabled", SetSampleEnabled, set_enabled),
                                      ("get_result", GetResult, get_result), ("new_session", Trigger, new_session)):
            self.services.append(self.node.create_service(kind, "~/" + name, guarded(function), callback_group=self.group))
        publisher = self.node.create_publisher(DiagnosticArray, "~/status", 1)
        def status():
            message = DiagnosticArray()
            message.header.stamp = self.node.get_clock().now().to_msg()
            state = DiagnosticStatus()
            state.name, state.message = "handeye", "按需读取；未自动检测停稳"
            state.level = DiagnosticStatus.OK
            state.values = [KeyValue(key="dataset_revision", value=str(controller.dataset.revision)),
                            KeyValue(key="active_job", value=str(controller.active_job or "")),
                            KeyValue(key="robot_read_policy", value="capture_only")]
            message.status = [state]
            publisher.publish(message)
        self.timer = self.node.create_timer(1.0, status, callback_group=self.group)

    def close(self):
        import rclpy
        self._stop.set()
        self.executor.shutdown(timeout_sec=2)
        self.thread.join(timeout=2)
        closer = getattr(self.pose_source, "close", None)
        if closer:
            closer()
        self.node.destroy_node()
        if self.context.ok():
            rclpy.shutdown(context=self.context)


def fill_transform(message, T, parent, child, stamp=None):
    from scipy.spatial.transform import Rotation
    message.header.frame_id, message.child_frame_id = parent, child
    if stamp is not None:
        message.header.stamp = stamp
    t = T[:3, 3]
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    message.transform.translation.x, message.transform.translation.y, message.transform.translation.z = map(float, t)
    message.transform.rotation.x, message.transform.rotation.y, message.transform.rotation.z, message.transform.rotation.w = map(float, q)

