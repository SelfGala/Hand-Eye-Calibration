from copy import deepcopy
from threading import Event
from concurrent.futures import Future
from types import SimpleNamespace as NS
import time
import numpy as np
import pytest
from handeye_core.config import load_config
from handeye_core.controller import Controller
from handeye_core.dataset import Dataset
from handeye_core.errors import HandeyeError
from handeye_core.models import Board
from handeye_core.ros_runtime import ServicePoseSource, decode_transform
from handeye_core.synthetic import DemoRig, create_dataset
from handeye_core.solver import solve
from handeye_core.transforms import inverse


def wait_job(controller, job_id):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        job = controller.get_job(job_id)
        if job["state"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.005)
    raise AssertionError("job did not finish")


def rig_controller(tmp_path):
    config = load_config()
    config["storage"]["root"] = str(tmp_path)
    rig = DemoRig(Board(**config["board"]))
    dataset = Dataset.create(tmp_path, config, rig.model)
    return Controller(config, dataset, rig, rig), rig


def test_preview_and_capture_have_no_stability_gate_or_extra_reads(tmp_path):
    controller, rig = rig_controller(tmp_path)
    try:
        for _ in range(10):
            rig.latest()
        assert rig.query_count == 0
        # Source says moving and quality warning threshold is intentionally exceeded.
        read = rig.read
        def moving(*args):
            value = read(*args)
            value.metadata.update(has_motion_state=True, is_moving=True)
            return value
        rig.read = moving
        controller.config["quality"]["warning_pnp_rmse_px"] = -1
        for i in range(3):
            job = wait_job(controller, controller.capture())
            assert job["state"] == "succeeded", job
            assert job["pose_request_count"] == 1
        assert rig.query_count == 3
        revision, samples = controller.dataset.snapshot()
        assert revision == 3 and len(samples) == 3
        assert samples[0]["transaction"]["stability_check"] == "manual_hold"
        assert samples[0]["warnings"]
        assert not any("settle" in k or "stable" in k for k in controller.config["sampling"])
    finally:
        controller.close()
        controller.pool.shutdown(wait=True)


def test_cancellation_does_not_commit_a_late_capture(tmp_path):
    controller, rig = rig_controller(tmp_path)
    entered, release = Event(), Event()
    original = rig.read
    def held(*args):
        reading = original(*args)
        entered.set()
        release.wait(2)
        return reading
    rig.read = held
    try:
        job_id = controller.capture()
        assert entered.wait(2)
        with pytest.raises(HandeyeError, match="已有任务"):
            controller.capture()
        controller.cancel()
        release.set()
        assert wait_job(controller, job_id)["state"] == "cancelled"
        assert controller.dataset.snapshot()[1] == []
        rig.read = original
        job = wait_job(controller, controller.capture())
        assert job["state"] == "succeeded" and job["pose_request_count"] == 1
        assert rig.query_count == 2
    finally:
        release.set()
        controller.close()
        controller.pool.shutdown(wait=True)


def test_offline_roundtrip_revision_and_original_record_are_preserved(tmp_path):
    config = load_config()
    dataset, _ = create_dataset(tmp_path, config, 15, 3)
    controller = Controller(config, dataset)
    try:
        job = wait_job(controller, controller.compute(["PARK"]))
        assert job["state"] == "succeeded", job
        assert job["pose_request_count"] == 0
        result = controller.get_result()
        sample_id = dataset.snapshot()[1][0]["sample_id"]
        original = (dataset.path / "samples" / sample_id / "sample.json").read_bytes()
        controller.flip_sample(sample_id)
        assert controller.get_result()["stale"]
        controller.flip_sample(sample_id)
        assert (dataset.path / "samples" / sample_id / "sample.json").read_bytes() == original
        reopened = Dataset(dataset.path)
        revision, samples = reopened.snapshot()
        recovered = solve(samples, reopened.camera)
        np.testing.assert_allclose(result["T_parent_child"], recovered["T_parent_child"], atol=1e-6)
        assert revision == 20
        folder = dataset.path / "results" / result["result_id"]
        assert all((folder / p).is_file() for p in ("calibration.yaml", "report.json", "report.csv", "report.txt"))
    finally:
        controller.close()


def message(T=None, parent="base_link", child="r_end_link"):
    return NS(header=NS(frame_id=parent, stamp=NS(sec=10, nanosec=20)), child_frame_id=child,
              transform=NS(translation=NS(x=0.1, y=0.2, z=0.3), rotation=NS(x=0, y=0, z=0, w=1)))


def test_ros_frame_direction_is_explicit():
    config = load_config()
    T, header = decode_transform(message(), config)
    with pytest.raises(HandeyeError):
        decode_transform(message(parent="r_end_link", child="base_link"), config)
    config["robot"]["allow_inverted_input"] = True
    recovered, header = decode_transform(message(parent="r_end_link", child="base_link"), config)
    np.testing.assert_allclose(recovered, inverse(T))
    assert header["input_inverted"]


def test_service_timeout_has_one_request_and_late_response_cannot_cross_capture():
    class Client:
        def __init__(self):
            self.requests, self.removed = [], []
            self.respond = False
        def service_is_ready(self):
            return True
        def call_async(self, request):
            self.requests.append(request)
            future = Future()
            if self.respond:
                future.set_result(NS(success=True, request_id=request.request_id, pose=message(),
                    query_started_at=NS(sec=10, nanosec=0), query_finished_at=NS(sec=10, nanosec=20),
                    timestamp_kind="host_query_midpoint", source_kind="sdk_on_request",
                    source_age_known=False, has_motion_state=False, is_moving=False,
                    sdk_call_count_known=True, sdk_call_count=1))
            else:
                future.set_running_or_notify_cancel()  # Simulate a request already executing remotely.
                self.late = future
            return future
        def remove_pending_request(self, future):
            self.removed.append(future)
    source = ServicePoseSource.__new__(ServicePoseSource)
    source.config = load_config()
    source.service_type = NS(Request=NS)
    source.client = Client()
    count = []
    with pytest.raises(HandeyeError, match="超时"):
        source.read("first", 0.01, Event(), lambda: count.append(1))
    assert len(count) == len(source.client.requests) == 1
    source.client.respond = True
    second = source.read("second", 0.5, Event(), lambda: count.append(1))
    source.client.late.set_result(NS(request_id="first"))
    assert second.metadata["request_id"] == "second"
    assert len(source.client.requests) == len(source.client.removed) == 2

