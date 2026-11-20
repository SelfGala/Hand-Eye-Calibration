from concurrent.futures import Future
from threading import Event, Timer
from types import SimpleNamespace as NS
import time
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from handeye_core.auto_motion import image_drift_deg, image_margin_deg, pivot_distance, sphere_targets
from handeye_core.config import load_config
from handeye_core.controller import Controller
from handeye_core.dataset import Dataset
from handeye_core.detection import from_corners, project
from handeye_core.errors import HandeyeError
from handeye_core.models import Board
from handeye_core import ros_runtime
from handeye_core.ros_runtime import MoveItMover, joint_limits_from_urdf, joint_margin_violation, slow_down
from handeye_core.solver import solve
from handeye_core.synthetic import DemoRig, pose_at, scene
from handeye_core.transforms import inverse, rotation_error_deg, transform


def wait_job(controller, job_id):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        job = controller.get_job(job_id)
        if job["state"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.005)
    raise AssertionError("job did not finish")


class Arm:
    """The demo rig as the mover; chosen move calls fail with a given error code."""
    def __init__(self, rig, failures):
        self.rig, self.failures, self.calls = rig, failures, 0

    def move(self, target, cancel):
        self.calls += 1
        if self.calls in self.failures:
            raise HandeyeError(self.failures[self.calls], "模拟")
        self.rig.move(target, cancel)


def auto_controller(tmp_path, points, failures=None, center_offset=(0, 0, 0)):
    config = load_config()
    config["storage"]["root"] = str(tmp_path)
    rig = DemoRig(Board(**config["board"]))
    arm = Arm(rig, failures or {})
    options = {**config["auto_motion"], "points": points,
               "board_center_m": (rig.board_center() + center_offset).tolist()}
    controller = Controller(config, Dataset.create(tmp_path, config, rig.model), rig, rig, arm, options)
    return controller, rig, arm


def close(controller):
    controller.close()
    controller.pool.shutdown(wait=True)


def test_targets_swing_around_the_board_centre_when_the_end_may_travel_that_far():
    seed = transform(Rotation.from_euler("xyz", [0.3, -1.2, 0.5]).as_matrix(), [-0.2, 0.5, 0.1])
    center = np.array([0.0, 1.5, 1.0])
    targets = sphere_targets(seed, center, 30, 20, 15, 1.0)
    assert len(targets) == 30
    view = center - seed[:3, 3]
    radius = np.linalg.norm(view)
    centre_in_end = inverse(seed) @ np.append(center, 1)
    tilts = []
    for target in targets:
        # The camera is rigid on the end link, so this is what keeps the board centred in the image.
        np.testing.assert_allclose(inverse(target) @ np.append(center, 1), centre_in_end, atol=1e-9)
        tilts.append(np.degrees(np.arccos(np.dot(center - target[:3, 3], view) / radius ** 2)))
    assert max(tilts) == pytest.approx(20) and tilts == sorted(tilts)
    rolls = [rotation_error_deg(seed[:3, :3], target[:3, :3]) for target in targets]
    assert max(rolls) > 20  # roll about the view axis adds to the tilt


def test_a_far_board_is_swung_around_a_nearer_pivot_to_bound_the_travel():
    seed = transform(Rotation.from_euler("xyz", [0.3, -1.2, 0.5]).as_matrix(), [-0.2, 0.5, 0.1])
    center = np.array([0.0, 1.5, 1.0])
    radius = np.linalg.norm(center - seed[:3, 3])
    targets = sphere_targets(seed, center, 30, 20, 15, 0.15)
    assert max(np.linalg.norm(target[:3, 3] - seed[:3, 3]) for target in targets) == pytest.approx(0.15)
    assert max(rotation_error_deg(seed[:3, :3], target[:3, :3]) for target in targets) > 20
    # The price: the board centre moves in the image, as seen from the end link.
    before = (inverse(seed) @ np.append(center, 1))[:3]
    drifts = []
    for target in targets:
        after = (inverse(target) @ np.append(center, 1))[:3]
        drifts.append(np.degrees(np.arccos(before @ after / np.linalg.norm(before) / np.linalg.norm(after))))
    pivot = pivot_distance(radius, 20, 0.15)
    assert pivot == pytest.approx(0.15 / (2 * np.sin(np.radians(10)))) and pivot < radius
    assert max(drifts) == pytest.approx(image_drift_deg(radius, pivot, 20))
    assert image_drift_deg(radius, pivot_distance(radius, 20, 1.0), 20) == pytest.approx(0)
    steps = [np.linalg.norm(b[:3, 3] - a[:3, 3]) for a, b in zip([seed] + targets, targets)]
    assert max(steps) < 0.5 * radius * np.radians(20)  # neighbours, never across the whole cap


def test_capture_moves_to_the_next_point_and_skips_unreachable_ones(tmp_path):
    controller, rig, arm = auto_controller(tmp_path, 4, {2: "UNREACHABLE"})
    try:
        job = wait_job(controller, controller.auto_start())
        assert job["state"] == "succeeded", job
        targets = controller.auto["targets"]
        assert arm.calls == 0  # nothing moves until a capture succeeds
        job = wait_job(controller, controller.capture())
        assert job["state"] == "succeeded" and "已到第 1/4 点" in job["message"], job
        job = wait_job(controller, controller.capture())
        assert "已到第 3/4 点，跳过了 1 个到不了的点" in job["message"], job
        np.testing.assert_allclose(controller.dataset.snapshot()[1][1]["G"], targets[0], atol=1e-9)
        np.testing.assert_allclose(rig._moved_to, targets[2])
        assert "已到第 4/4 点" in wait_job(controller, controller.capture())["message"]
        job = wait_job(controller, controller.capture())
        assert job["state"] == "succeeded" and "自动换位结束" in job["message"], job
        assert controller.auto is None and arm.calls == 4
        assert len(controller.dataset.snapshot()[1]) == 4
        assert "可以移动到下一姿态" in wait_job(controller, controller.capture())["message"]
        assert arm.calls == 4
    finally:
        close(controller)


def test_failed_capture_does_not_move(tmp_path, monkeypatch):
    controller, rig, arm = auto_controller(tmp_path, 4)
    try:
        assert wait_job(controller, controller.auto_start())["state"] == "succeeded"
        def boom(*args, **kwargs):
            raise HandeyeError("CHESSBOARD_INCOMPLETE", "模拟识别失败")
        monkeypatch.setattr("handeye_core.controller.detect", boom)
        assert wait_job(controller, controller.capture())["state"] == "failed"
        assert arm.calls == 0 and controller.auto["index"] == 0
    finally:
        close(controller)


def test_execution_failure_stops_until_the_operator_continues(tmp_path):
    controller, rig, arm = auto_controller(tmp_path, 4, {1: "MOVE_FAILED"})
    try:
        assert wait_job(controller, controller.auto_start())["state"] == "succeeded"
        job = wait_job(controller, controller.capture())
        assert job["state"] == "failed" and job["error_code"] == "MOVE_FAILED", job
        assert job["message"].startswith("已保存") and job["sample_id"]
        assert len(controller.dataset.snapshot()[1]) == 1
        assert arm.calls == 1  # no automatic retry or skip after a failed execution
        job = wait_job(controller, controller.auto_next())
        assert job["state"] == "succeeded" and "已到第 2/4 点" in job["message"], job
        controller.auto_stop()
        assert wait_job(controller, controller.auto_next())["error_code"] == "AUTO_OFF"
        assert arm.calls == 2
    finally:
        close(controller)


def test_start_refuses_a_board_centre_that_disagrees_with_the_camera(tmp_path):
    controller, rig, arm = auto_controller(tmp_path, 4, center_offset=(1.5, 0, 0))
    try:
        job = wait_job(controller, controller.auto_start())
        assert job["state"] == "failed" and job["error_code"] == "AUTO_BOARD_CENTER", job
        assert controller.auto is None
    finally:
        close(controller)


def test_image_margin_is_the_angle_from_the_outermost_corner_to_the_image_edge():
    camera = NS(K=[[500, 0, 500], [0, 500, 400], [0, 0, 1]], width=1000, height=800)
    board = NS(pattern_size=(3, 2))
    corners = np.array([[450, 375], [500, 375], [550, 375], [450, 425], [500, 425], [550, 425]], float)
    # With the squares around the inner corners the board spans 400..600 by 325..475 pixels.
    upright = np.degrees(np.arctan(400 / 500) - np.arctan(75 / 500))
    assert image_margin_deg(corners, board, camera, 0) == pytest.approx(upright)
    assert image_margin_deg(corners + [300, 0], board, camera, 0) == pytest.approx(np.degrees(np.arctan(1) - np.arctan(400 / 500)))
    quarter_turn = np.degrees(np.arctan(400 / 500) - np.arctan(100 / 500))  # the long side now runs up the image
    assert image_margin_deg(corners, board, camera, 90) == pytest.approx(quarter_turn)
    assert quarter_turn < upright


def test_poses_on_the_sphere_determine_the_hand_eye_transform():
    board = Board()
    X, Y, camera = scene()
    seed = Y @ inverse(pose_at(0, board)) @ inverse(X)
    center = Y[:3, :3] @ board.object_points().mean(axis=0) + Y[:3, 3]
    samples = []
    for i, G in enumerate([seed] + sphere_targets(seed, center, 15, 20, 20, 0.15)):
        pixels = project(board.object_points(), inverse(X) @ inverse(G) @ Y, camera)
        assert (pixels > 0).all() and (pixels < [camera.width, camera.height]).all()  # board stays in view
        samples.append({"sample_id": str(i), "role": "train", "enabled": True, "G": G.tolist(),
                        "analysis": from_corners(pixels, board, camera)})
    estimate = np.asarray(solve(samples, camera, "PARK")["T_parent_child"])
    assert np.linalg.norm(estimate[:3, 3] - X[:3, 3]) * 1000 < 0.01
    assert rotation_error_deg(estimate[:3, :3], X[:3, :3]) < 0.001


def point(seconds, velocity):
    return NS(time_from_start=NS(sec=int(seconds), nanosec=round(seconds % 1 * 1e9)),
              velocities=[velocity], accelerations=[2.0])


def test_slow_down_stretches_time_and_drops_points_that_do_not_advance():
    trajectory = NS(points=[point(0, 0.0), point(0, 0.0), point(0.25, 1.0), point(1.5, 0.0)])
    assert slow_down(trajectory, 0.1) == pytest.approx(15.0)
    assert [(p.time_from_start.sec, p.time_from_start.nanosec) for p in trajectory.points] == [
        (0, 0), (2, 500_000_000), (15, 0)]
    assert trajectory.points[1].velocities == [pytest.approx(0.1)]
    assert trajectory.points[1].accelerations == [pytest.approx(0.02)]


def done(value):
    future = Future()
    future.set_result(value)
    return future


def moveit_mover(fraction=1.0, result=None):
    """MoveItMover wired to stand-ins for the move_group service and action."""
    log = NS(requests=[], goals=[], cancels=0, stops=0, listeners=1, obeys_stop=True, result=result or Future())
    def cancel_goal():
        log.cancels += 1  # Humble's move_group keeps executing after this
    def stop(message):
        assert message.data == "stop"
        log.stops += 1
        if log.obeys_stop and not log.result.done():
            log.result.set_result(NS(result=NS(error_code=NS(val=-7))))
    def plan(request):
        log.requests.append(request)
        trajectory = NS(points=[point(0, 0.0), point(0.5, 1.0), point(1.0, 0.0)])
        return done(NS(error_code=NS(val=1), fraction=fraction, solution=NS(joint_trajectory=trajectory)))
    def send(goal):
        log.goals.append(goal)
        return done(NS(accepted=True, get_result_async=lambda: log.result, cancel_goal_async=cancel_goal))
    mover = MoveItMover(None, {"base": "base_link", "end": "l_end_link"},
                        {"move_group": "left_arm", "velocity_scale": 0.1, "joint_limit_margin_rad": 0.1}, None)
    mover.joint_limits = {}
    mover.pose_type = lambda: NS(position=NS(), orientation=NS())
    mover.plan_type = lambda: NS(header=NS(), start_state=NS())
    mover.goal_type = mover.stop_type = NS
    mover.stop_publisher = NS(publish=stop, get_subscription_count=lambda: log.listeners)
    mover.plan_client = NS(wait_for_service=lambda timeout_sec: True, call_async=plan,
                           remove_pending_request=lambda future: None)
    mover.execute_client = NS(wait_for_server=lambda timeout_sec: True, send_goal_async=send)
    return mover, log


def test_mover_plans_for_the_configured_arm_and_executes_slowly():
    mover, log = moveit_mover(result=done(NS(result=NS(error_code=NS(val=1)))))
    mover.move(transform(t=[0.1, 0.5, 0.2]), Event())
    request = log.requests[0]
    assert (request.header.frame_id, request.group_name, request.link_name) == ("base_link", "left_arm", "l_end_link")
    assert request.start_state.is_diff and request.avoid_collisions and request.jump_threshold > 0
    assert (request.waypoints[0].position.y, request.waypoints[0].orientation.w) == (0.5, 1.0)
    sent = log.goals[0].trajectory.joint_trajectory.points
    assert sent[-1].time_from_start.sec == 10 and sent[1].velocities == [pytest.approx(0.1)]


def test_auto_mode_does_not_start_without_move_group(tmp_path):
    controller, rig, arm = auto_controller(tmp_path, 3)
    def connect():
        raise HandeyeError("MOVE_GROUP_UNAVAILABLE", "模拟")
    arm.connect = connect
    try:
        assert wait_job(controller, controller.auto_start())["error_code"] == "MOVE_GROUP_UNAVAILABLE"
        assert controller.auto is None
    finally:
        close(controller)


def test_mover_reports_a_partial_path_as_unreachable_without_moving():
    mover, log = moveit_mover(fraction=0.6)
    with pytest.raises(HandeyeError) as error:
        mover.move(transform(), Event())
    assert error.value.code == "UNREACHABLE" and log.goals == []


def test_mover_reports_a_failed_execution():
    mover, log = moveit_mover(result=done(NS(result=NS(error_code=NS(val=-4)))))
    with pytest.raises(HandeyeError) as error:
        mover.move(transform(), Event())
    assert error.value.code == "MOVE_FAILED"


def cancelled_move(mover):
    cancel = Event()
    timer = Timer(0.1, cancel.set)
    timer.start()
    try:
        with pytest.raises(HandeyeError) as error:
            mover.move(transform(), cancel)
    finally:
        timer.cancel()
    return error.value.code


def test_cancel_during_execution_stops_the_arm():
    mover, log = moveit_mover()
    assert cancelled_move(mover) == "CANCELLED"
    assert log.stops == 1 and log.cancels == 1


def test_a_stop_that_move_group_does_not_confirm_is_reported(monkeypatch):
    monkeypatch.setattr(ros_runtime, "STOP_CONFIRM_SECONDS", 0.3)
    mover, log = moveit_mover()
    log.obeys_stop = False
    assert cancelled_move(mover) == "STOP_UNCONFIRMED"
    assert log.stops > 1


def test_nothing_moves_when_move_group_cannot_be_told_to_stop(monkeypatch):
    monkeypatch.setattr(ros_runtime, "CONNECT_SECONDS", 0.1)
    mover, log = moveit_mover()
    log.listeners = 0
    with pytest.raises(HandeyeError) as error:
        mover.move(transform(), Event())
    assert error.value.code == "MOVE_GROUP_UNAVAILABLE" and log.requests == [] and log.goals == []


def test_joint_command_stays_a_tenth_inside_the_urdf_limits():
    limits = joint_limits_from_urdf("""
        <robot name="arm">
          <joint name="shoulder" type="revolute"><limit lower="-0.8" upper="0.8"/></joint>
          <joint name="wrist" type="continuous"/>
        </robot>
    """)
    assert limits == {"shoulder": (-0.8, 0.8)}
    outside = NS(joint_names=["shoulder", "wrist"], points=[
        NS(positions=[0.2, 1.2]),
        NS(positions=[0.8, 1.2]),
    ])
    assert "0.800" in joint_margin_violation(outside, limits, 0.1)
    assert outside.points[1].positions == [0.8, 1.2]  # refused, not rewritten
    inside = NS(joint_names=["shoulder"], points=[NS(positions=[0.3]), NS(positions=[0.2])])
    assert joint_margin_violation(inside, limits, 0.1) is None
    # Already closer to the stop than the margin: may move back, not further out.
    parked = NS(joint_names=["shoulder"], points=[NS(positions=[0.75]), NS(positions=[0.5])])
    assert joint_margin_violation(parked, limits, 0.1) is None
    further = NS(joint_names=["shoulder"], points=[NS(positions=[0.75]), NS(positions=[0.78])])
    assert joint_margin_violation(further, limits, 0.1)


def test_stamp_a_little_ahead_of_this_clock_is_still_fresh():
    now = 1_000_000_000_000
    fresh, age_ns = ros_runtime._fresh_stamp(now, now + 80_000_000, now, 2.0)
    assert fresh and age_ns == -80_000_000
    assert not ros_runtime._fresh_stamp(now, now + 2_000_000_000, now, 2.0)[0]
    assert not ros_runtime._fresh_stamp(now, now - 100_000_000, now, 2.0)[0]
