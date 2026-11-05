from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from threading import Event, RLock
import time
import json
import numpy as np
from .dataset import Dataset, new_id, read_image
from .detection import detect, from_corners
from .errors import HandeyeError
from .solver import solve
from .transforms import checked


@dataclass
class Job:
    job_id: str
    kind: str
    state: str = "queued"
    message: str = ""
    error_code: str = ""
    sample_id: str = ""
    result_ids: list = field(default_factory=list)
    pose_request_count: int = 0
    method_errors: dict = field(default_factory=dict)


class Controller:
    def __init__(self, config, dataset, camera=None, pose_source=None):
        self.config = deepcopy(config)
        self.dataset = dataset
        self.camera = camera
        self.pose_source = pose_source
        self.lock = RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="handeye-job")
        self.jobs = {}
        self.results = {}
        self.latest_result_id = ""
        self.active_job = None
        self.cancel_event = Event()
        self.closed = False
        for path in sorted(self.dataset.path.glob("results/*/report.json")):
            if path.parent.name.startswith(".pending_"):
                continue
            with path.open(encoding="utf-8") as f:
                result = json.load(f)
            self.results[result["result_id"]] = result
            self.latest_result_id = result["result_id"]

    def _submit(self, kind, function):
        with self.lock:
            if self.closed or self.active_job:
                raise HandeyeError("BUSY", "已有任务在执行或程序正在关闭")
            job = Job(new_id("job"), kind)
            self.jobs[job.job_id] = job
            # Bound in-memory history; results themselves are persisted on disk.
            if len(self.jobs) > 200:
                self.jobs.pop(next(iter(self.jobs)))
            self.active_job = job.job_id
            self.cancel_event = Event()
            self.pool.submit(self._run, job, function, self.cancel_event)
            return job.job_id

    def _run(self, job, function, cancel):
        try:
            with self.lock:
                job.state = "running"
            function(job, cancel)
            with self.lock:
                job.state = "succeeded"
        except Exception as exc:
            with self.lock:
                job.state = "cancelled" if isinstance(exc, HandeyeError) and exc.code == "CANCELLED" else "failed"
                job.error_code = getattr(exc, "code", "INTERNAL_ERROR")
                job.message = str(exc)
        finally:
            with self.lock:
                self.active_job = None

    @staticmethod
    def _cancelled(cancel):
        if cancel.is_set():
            raise HandeyeError("CANCELLED", "任务已取消")

    def capture(self, role="train", label="", flip=False):
        if role not in ("train", "validation"):
            raise HandeyeError("ROLE", "角色必须为 train 或 validation")
        if self.camera is None or self.pose_source is None:
            raise HandeyeError("CAPTURE_UNAVAILABLE", "当前离线会话没有相机或机器人输入")
        def execute(job, cancel):
            start_ns = time.monotonic_ns()
            request_id = new_id("pose")
            job.message = "读取本次位姿并采集图像"
            # No countdown, stability estimation, quality pre-gate, polling or retry.
            reading = self.pose_source.read(request_id, self.config["robot"]["query_timeout_seconds"],
                                             cancel, lambda: self._count(job))
            G = checked(reading.G)
            self._cancelled(cancel)
            response_ns = time.monotonic_ns()
            elapsed = (response_ns - start_ns) / 1e9
            remaining = self.config["sampling"]["capture_timeout_seconds"] - elapsed
            if remaining <= 0:
                raise HandeyeError("CAPTURE_TIMEOUT", "本轮采样超时")
            frame = self.camera.fresh(response_ns, remaining, cancel)
            analysis = detect(frame.image, self.dataset.board, self.dataset.camera, flip)
            warnings = []
            if analysis["pnp_rmse_px"] > self.config["quality"]["warning_pnp_rmse_px"]:
                warnings.append("PnP 重投影误差较大；已保存，可手动停用")
            if analysis["hull_area_ratio"] < self.config["quality"]["warning_hull_area_ratio"]:
                warnings.append("棋盘覆盖率较低；已保存，可调整取景")
            if reading.metadata.get("has_motion_state") and reading.metadata.get("is_moving"):
                warnings.append("来源标记运动中；仅记录，不阻止拍照")
            self._cancelled(cancel)
            record = {"schema_version": 2, "role": role, "label": label, "G": G.tolist(),
                      "analysis": analysis, "robot": reading.metadata, "camera_frame": frame.metadata(),
                      "transaction": {"request_id": request_id, "started_monotonic_ns": start_ns,
                          "pose_returned_monotonic_ns": response_ns, "pose_request_count": job.pose_request_count,
                          "sdk_call_count": reading.metadata.get("sdk_call_count"),
                          "sdk_count_unavailable_reason": reading.metadata.get("sdk_count_unavailable_reason"),
                          "pairing_mode": "manual_stationary_snapshot", "stability_check": "manual_hold",
                          "shared_clock_delta_ms": None,
                          "local_response_to_frame_ms": (frame.host_monotonic_ns - response_ns) / 1e6},
                      "warnings": warnings}
            # Cancel is serialized with the final commit, so cancellation cannot create a later sample.
            with self.lock:
                self._cancelled(cancel)
                job.sample_id = self.dataset.add(frame.image, record)
            job.message = "已保存，可以移动到下一姿态" + ("；" + "；".join(warnings) if warnings else "")
        return self._submit("capture", execute)

    def _count(self, job):
        with self.lock:
            job.pose_request_count += 1

    def compute(self, methods=None):
        methods = list(methods or self.config["calibration"]["algorithms"])
        def execute(job, cancel):
            revision, samples = self.dataset.snapshot()
            for method in methods:
                self._cancelled(cancel)
                try:
                    result = solve(samples, self.dataset.camera, method, self.config["validation"])
                except Exception as exc:
                    job.method_errors[method] = str(exc)
                    continue
                self._cancelled(cancel)
                result = self.dataset.save_result(result, revision)
                with self.lock:
                    self.results[result["result_id"]] = result
                    if not job.result_ids:
                        self.latest_result_id = result["result_id"]
                    job.result_ids.append(result["result_id"])
            if not job.result_ids:
                raise HandeyeError("COMPUTE_FAILED", str(job.method_errors))
            job.message = "计算完成" + (f"；部分算法失败：{job.method_errors}" if job.method_errors else "")
        return self._submit("compute", execute)

    def get_job(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise HandeyeError("JOB_NOT_FOUND", job_id)
            return deepcopy(asdict(self.jobs[job_id]))

    def get_result(self, result_id=""):
        with self.lock:
            result_id = result_id or self.latest_result_id
            if result_id not in self.results:
                raise HandeyeError("RESULT_NOT_FOUND", "尚未计算结果或结果 ID 不存在")
            result = deepcopy(self.results[result_id])
            result["stale"] = result["dataset_revision"] != self.dataset.revision
            return result

    def _editable(self):
        if self.active_job or self.closed:
            raise HandeyeError("BUSY", "请等待当前任务完成")

    def set_sample(self, sample_id, **changes):
        with self.lock:
            self._editable()
            self.dataset.update(sample_id, **changes)

    def flip_sample(self, sample_id):
        with self.lock:
            self._editable()
            _, samples = self.dataset.snapshot()
            sample = next(s for s in samples if s["sample_id"] == sample_id)
            old = sample["analysis"]
            analysis = from_corners(old["raw_corners"], self.dataset.board, self.dataset.camera,
                                     not old.get("flip_180", False))
            if "sharpness_laplacian_var" in old:
                analysis["sharpness_laplacian_var"] = old["sharpness_laplacian_var"]
            self.dataset.replace_analyses({sample_id: analysis}, "manual_flip_180")

    def new_session(self, board_config=None):
        with self.lock:
            self._editable()
            if board_config:
                self.config["board"] = board_config
            self.dataset = Dataset.create(self.config["storage"]["root"], self.config, self.dataset.camera,
                                           self.dataset.session.get("environment"))
            self.results = {}
            self.latest_result_id = ""

    def cancel(self):
        with self.lock:
            self.cancel_event.set()

    def close(self):
        with self.lock:
            self.closed = True
            self.cancel_event.set()
        self.pool.shutdown(wait=False, cancel_futures=True)

