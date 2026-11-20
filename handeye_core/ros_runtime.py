"""ROS is imported only on a hardware/ROS entry path. Executor runs independently of Qt."""
from threading import Event, Lock, Thread
import json
import time
import numpy as np
from .errors import HandeyeError
from .models import PoseReading
from .transforms import from_pose, inverse, pose_components

PLAN_SERVICE = "/compute_cartesian_path"
EXECUTE_ACTION = "/execute_trajectory"
STOP_TOPIC = "/trajectory_execution_event"
MOVEIT_SUCCESS = 1
CONNECT_SECONDS = 3.0
STOP_CONFIRM_SECONDS = 3.0


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


def _wait(future, cancel, timeout, what):
    done = Event()
    future.add_done_callback(lambda _: done.set())
    deadline = time.monotonic() + timeout
    while not done.wait(0.02):
        if cancel.is_set():
            raise HandeyeError("CANCELLED", "已取消")
        if time.monotonic() >= deadline:
            raise HandeyeError("MOVE_TIMEOUT", f"{what}超时")
    return future.result()


def slow_down(trajectory, scale):
    """Stretch a joint trajectory in time and return its new duration in seconds.

    move_group times Cartesian paths at full speed and takes no scaling factor in the request.
    Points that do not advance in time are dropped; controllers reject them.
    """
    points, last = [], -1.0
    for point in trajectory.points:
        t = (point.time_from_start.sec + point.time_from_start.nanosec * 1e-9) / scale
        if t <= last:
            continue
        last = t
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(round(t * 1e9), 1_000_000_000)
        point.velocities = [v * scale for v in point.velocities]
        point.accelerations = [a * scale * scale for a in point.accelerations]
        points.append(point)
    trajectory.points = points
    return max(last, 0.0)


class MoveItMover:
    """Moves the end link through move_group on the robot computer.

    A straight Cartesian path is planned from the current state, then executed. A target the
    path cannot fully reach raises UNREACHABLE before anything moves.

    Humble's move_group does not stop when an ExecuteTrajectory goal is cancelled. What stops the
    arm is "stop" on the execution event topic, so no motion starts unless move_group listens there.
    """
    def __init__(self, node, frames, options, group):
        self.node, self.frames, self.options, self.group = node, frames, options, group
        self.plan_client = self.execute_client = None

    def connect(self):
        if self.plan_client is None:
            try:
                from moveit_msgs.action import ExecuteTrajectory
                from moveit_msgs.srv import GetCartesianPath
            except ImportError as exc:
                raise HandeyeError("MOVEIT_MSGS", "本机没有 moveit_msgs，请安装 ros-$ROS_DISTRO-moveit-msgs") from exc
            from geometry_msgs.msg import Pose
            from rclpy.action import ActionClient
            from std_msgs.msg import String
            self.pose_type, self.plan_type, self.goal_type = Pose, GetCartesianPath.Request, ExecuteTrajectory.Goal
            self.stop_type = String
            self.stop_publisher = self.node.create_publisher(String, STOP_TOPIC, 10)
            self.plan_client = self.node.create_client(GetCartesianPath, PLAN_SERVICE, callback_group=self.group)
            self.execute_client = ActionClient(self.node, ExecuteTrajectory, EXECUTE_ACTION, callback_group=self.group)
        deadline = time.monotonic() + CONNECT_SECONDS
        ready = (self.plan_client.wait_for_service(timeout_sec=CONNECT_SECONDS)
                 and self.execute_client.wait_for_server(timeout_sec=CONNECT_SECONDS))
        while ready and self.stop_publisher.get_subscription_count() == 0:
            ready = time.monotonic() < deadline
            time.sleep(0.02)
        if not ready:
            raise HandeyeError("MOVE_GROUP_UNAVAILABLE", f"找不到 move_group 的 {PLAN_SERVICE}、{EXECUTE_ACTION} 或 {STOP_TOPIC}")

    def _stop(self, handle, result):
        """Stop the arm and wait until move_group reports the execution over."""
        handle.cancel_goal_async()  # enough on MoveIt versions that honour it
        over = Event()
        result.add_done_callback(lambda _: over.set())
        deadline = time.monotonic() + STOP_CONFIRM_SECONDS
        while True:
            # Repeated: a stop that arrives before move_group starts executing an accepted goal is lost.
            self.stop_publisher.publish(self.stop_type(data="stop"))
            if over.wait(0.1):
                return
            if time.monotonic() >= deadline:
                raise HandeyeError("STOP_UNCONFIRMED", "已发出停止指令，但 move_group 没有确认停下；请立刻确认手臂状态，必要时按急停")

    def _stop_late_goal(self, sending):
        handle = None if sending.cancelled() or sending.exception() else sending.result()
        if handle is not None and handle.accepted:
            Thread(target=self._stop, args=(handle, handle.get_result_async()), daemon=True).start()

    def move(self, target, cancel):
        self.connect()
        pose = pose_components(target)
        waypoint = self.pose_type()
        waypoint.position.x, waypoint.position.y, waypoint.position.z = pose["translation_m"]
        (waypoint.orientation.x, waypoint.orientation.y,
         waypoint.orientation.z, waypoint.orientation.w) = pose["quaternion_xyzw"]
        request = self.plan_type()
        request.header.frame_id = self.frames["base"]
        request.start_state.is_diff = True  # start from the current state
        request.group_name = self.options["move_group"]
        request.link_name = self.frames["end"]
        request.waypoints = [waypoint]
        request.max_step = 0.01
        # The IK solver may return a far-away arm configuration; cut the path there instead of following it.
        request.jump_threshold = 5.0
        request.avoid_collisions = True
        planning = self.plan_client.call_async(request)
        try:
            plan = _wait(planning, cancel, 10.0, "路径规划")
        finally:
            self.plan_client.remove_pending_request(planning)
        if plan.error_code.val != MOVEIT_SUCCESS:
            raise HandeyeError("MOVE_FAILED", f"move_group 没有接受规划请求，错误码 {plan.error_code.val}")
        if plan.fraction < 0.999:
            raise HandeyeError("UNREACHABLE", f"直线路径只能走完 {plan.fraction:.0%}")
        duration = slow_down(plan.solution.joint_trajectory, self.options["velocity_scale"])
        if duration <= 0:
            raise HandeyeError("UNREACHABLE", "规划结果没有可执行的时间参数")
        goal = self.goal_type()
        goal.trajectory = plan.solution
        sending = self.execute_client.send_goal_async(goal)
        try:
            handle = _wait(sending, cancel, 5.0, "发送轨迹")
        except HandeyeError:
            # The goal can still be accepted after we stop waiting; it must not run then.
            sending.add_done_callback(self._stop_late_goal)
            raise
        if not handle.accepted:
            raise HandeyeError("MOVE_FAILED", "move_group 拒绝执行这条轨迹")
        finished = handle.get_result_async()
        try:
            result = _wait(finished, cancel, duration * 1.5 + 10.0, "轨迹执行").result
        except HandeyeError:
            self._stop(handle, finished)
            raise
        if result.error_code.val != MOVEIT_SUCCESS:
            raise HandeyeError("MOVE_FAILED", f"轨迹执行失败，MoveIt 错误码 {result.error_code.val}；先确认手臂状态")

    def close(self):
        if self.execute_client is not None:
            self.execute_client.destroy()


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
        self.mover = None
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

    def create_mover(self, options):
        self.mover = MoveItMover(self.node, self.config["frames"], options, self.group)
        return self.mover

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
        if self.mover:
            self.mover.close()
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

