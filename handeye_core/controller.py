from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from threading import Event, RLock
import time
import json
import numpy as np
from .auto_motion import image_drift_deg, image_margin_deg, pivot_distance, sphere_targets
from .dataset import Dataset, new_id, read_image, remember_session
from .detection import detect, from_corners
from .errors import HandeyeError
from .solver import solve
from .transforms import checked, tf_snapshot

# The camera sits within a few centimetres of the end link, so the configured board centre and
# the board the camera sees must be about equally far away. More than this means a wrong config.
BOARD_DISTANCE_TOLERANCE_M = 0.3


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
    def __init__(self, config, dataset, camera=None, pose_source=None, mover=None, auto_motion=None):
        self.config = deepcopy(config)
        self.dataset = dataset
        self.camera = camera
        self.pose_source = pose_source
        self.mover = mover
        self.auto_motion = deepcopy(auto_motion)
        self.auto = None  # {"targets", "index"} while automatic repositioning is on
        self.lock = RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="handeye-job")
        self.jobs = {}
        self.results = {}
        self.latest_result_id = ""
        self.active_job = None
        self.cancel_event = Event()
        self.closed = False
        self._camera_compatible(self.dataset)
        self.recover_pending()
        self._load_results()

    def _load_results(self):
        self.results = {}
        self.latest_result_id = ""
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
            payload = {"role": role, "label": label, "flip": bool(flip), "G": G.tolist(),
                       "tf": tf_snapshot(G, reading.metadata, self.config["frames"]),
                       "robot": reading.metadata, "camera_frame": frame.metadata(),
                       "transaction": {"request_id": request_id, "started_monotonic_ns": start_ns,
                           "pose_returned_monotonic_ns": response_ns, "pose_request_count": job.pose_request_count,
                           "sdk_call_count": reading.metadata.get("sdk_call_count"),
                           "sdk_count_unavailable_reason": reading.metadata.get("sdk_count_unavailable_reason"),
                           "pairing_mode": "manual_stationary_snapshot", "stability_check": "manual_hold",
                           "shared_clock_delta_ms": None,
                           "local_response_to_frame_ms": (frame.host_monotonic_ns - response_ns) / 1e6}}
            # Durable journal: image + arm pose + TF exist even if detection or the process stops.
            cache_id = self.dataset.stage_capture(frame.image, payload)
            try:
                self._cancelled(cancel)
                analysis = detect(frame.image, self.dataset.board, self.dataset.camera, flip)
                record = self._sample_record(payload, analysis)
                # Cancel is serialized with the final commit, so cancellation cannot create a later sample.
                with self.lock:
                    self._cancelled(cancel)
                    job.sample_id = self.dataset.add(frame.image, record)
                    self.dataset.finalize_cache(cache_id)
            except Exception as exc:
                self._keep_or_drop_cache(cache_id, exc)
                raise
            warnings = "；" + "；".join(record["warnings"]) if record["warnings"] else ""
            if self.auto is None:
                job.message = "已保存，可以移动到下一姿态" + warnings
            else:
                self._advance(job, cancel, "已保存" + warnings + "；")
        return self._submit("capture", execute)

    def auto_start(self):
        """Turn on automatic repositioning, with the current pose as its starting point."""
        if self.mover is None or self.camera is None or self.pose_source is None:
            raise HandeyeError("AUTO_UNAVAILABLE", "当前运行方式不能自动换位")
        def execute(job, cancel):
            options = self.auto_motion
            connect = getattr(self.mover, "connect", None)
            if connect:
                job.message = "连接 move_group"
                connect()  # a missing move_group should show now, not after the first capture
            job.message = "读取起点位姿，核对棋盘距离"
            reading = self.pose_source.read(new_id("pose"), self.config["robot"]["query_timeout_seconds"],
                                             cancel, lambda: self._count(job))
            seed = checked(reading.G)
            frame = self.camera.fresh(time.monotonic_ns(), self.config["sampling"]["capture_timeout_seconds"], cancel)
            board = self.dataset.board
            seen_board = detect(frame.image, board, self.dataset.camera)
            C = np.asarray(seen_board["C"])
            seen = np.linalg.norm(C[:3, :3] @ board.object_points().mean(axis=0) + C[:3, 3])
            center = np.asarray(options["board_center_m"], float)
            radius = np.linalg.norm(center - seed[:3, 3])
            if abs(radius - seen) > BOARD_DISTANCE_TOLERANCE_M:
                raise HandeyeError("AUTO_BOARD_CENTER", f"配置的棋盘中心离末端 {radius:.2f} m，相机实测棋盘在 {seen:.2f} m；"
                                                        "请核对 auto_motion.board_center_m")
            targets = sphere_targets(seed, center, options["points"], options["max_tilt_deg"],
                                     options["max_roll_deg"], options["max_travel_m"])
            pivot = pivot_distance(radius, options["max_tilt_deg"], options["max_travel_m"])
            drift = image_drift_deg(radius, pivot, options["max_tilt_deg"])
            margin = image_margin_deg(seen_board["corners"], board, self.dataset.camera, options["max_roll_deg"])
            with self.lock:
                self._cancelled(cancel)
                self.auto = {"targets": targets, "index": 0}
            job.message = (f"自动换位已开启：棋盘距末端 {radius:.2f} m，绕前方 {pivot:.2f} m 处转动，共 {len(targets)} 个点"
                           + (f"，棋盘在画面里最多偏移约 {drift:.0f}°" if drift > 1 else "")
                           + "。先在当前姿态拍照，成功后自动移动"
                           + (f"。注意：棋盘离画面边缘只有约 {margin:.0f}°，靠外的点可能拍不全；可把棋盘摆到画面中间，或调大 max_travel_m"
                              if drift > margin else ""))
        return self._submit("auto_start", execute)

    def auto_stop(self):
        with self.lock:
            self._editable()
            self.auto = None

    def auto_next(self):
        """Give up on the current point and move to the next one."""
        return self._submit("move", lambda job, cancel: self._advance(job, cancel))

    def _advance(self, job, cancel, done=""):
        """Move to the next target the arm can reach; targets it cannot reach are skipped."""
        skipped = 0
        while True:
            with self.lock:
                if self.auto is None:
                    raise HandeyeError("AUTO_OFF", "自动换位没有开启")
                targets, index = self.auto["targets"], self.auto["index"]
                if index == len(targets):
                    self.auto = None
                    job.message = done + (f"剩下 {skipped} 个点到不了；" if skipped else "") + "所有点已走完，自动换位结束"
                    return
                self.auto["index"] = index + 1
            job.message = f"{done}正在移动到第 {index + 1}/{len(targets)} 点"
            try:
                self.mover.move(targets[index], cancel)
            except HandeyeError as exc:
                if exc.code == "UNREACHABLE":
                    skipped += 1
                    continue
                raise HandeyeError(exc.code, f"{done}{exc}") from exc
            job.message = (f"{done}已到第 {index + 1}/{len(targets)} 点" + (f"，跳过了 {skipped} 个到不了的点" if skipped else "")
                           + "；停稳后拍照")
            return

    def _warnings(self, analysis, metadata):
        warnings = []
        if analysis["pnp_rmse_px"] > self.config["quality"]["warning_pnp_rmse_px"]:
            warnings.append("PnP 重投影误差较大；已保存，可手动停用")
        if analysis["hull_area_ratio"] < self.config["quality"]["warning_hull_area_ratio"]:
            warnings.append("棋盘覆盖率较低；已保存，可调整取景")
        if metadata.get("has_motion_state") and metadata.get("is_moving"):
            warnings.append("来源标记运动中；仅记录，不阻止拍照")
        return warnings

    def _sample_record(self, payload, analysis):
        return {"schema_version": 2, "role": payload["role"], "label": payload.get("label", ""),
                "G": payload["G"], "tf": payload["tf"], "analysis": analysis,
                "robot": payload["robot"], "camera_frame": payload["camera_frame"],
                "transaction": payload["transaction"], "warnings": self._warnings(analysis, payload["robot"])}

    def _keep_or_drop_cache(self, cache_id, exc):
        if isinstance(exc, HandeyeError) and exc.code == "CANCELLED":
            self.dataset.discard_cache(cache_id)
            return
        code = getattr(exc, "code", "INTERNAL_ERROR")
        self.dataset.mark_cache_error(cache_id, code, str(exc))
        raise HandeyeError(code, f"{exc}；原图、机械臂位姿和 TF 已写入缓存，可继续该会话恢复") from exc

    def recover_pending(self):
        """Commit cached captures that never became samples. Failed detections stay on disk."""
        recovered, failed = [], []
        for item in self.dataset.pending_captures():
            request_id = item.get("transaction", {}).get("request_id")
            existing = self.dataset.find_sample_by_request(request_id)
            if existing:
                self.dataset.finalize_cache(item["capture_id"])
                recovered.append({"capture_id": item["capture_id"], "sample_id": existing, "already_committed": True})
                continue
            try:
                image = read_image(item["image_path"])
                analysis = detect(image, self.dataset.board, self.dataset.camera, item.get("flip", False))
                sample_id = self.dataset.add(image, self._sample_record(item, analysis))
                self.dataset.finalize_cache(item["capture_id"])
                recovered.append({"capture_id": item["capture_id"], "sample_id": sample_id, "already_committed": False})
            except Exception as exc:
                self.dataset.mark_cache_error(item["capture_id"], getattr(exc, "code", "INTERNAL_ERROR"), str(exc))
                failed.append({"capture_id": item["capture_id"],
                               "error_code": getattr(exc, "code", "INTERNAL_ERROR"), "error": str(exc)})
        return {"recovered": recovered, "failed": failed}

    def _camera_compatible(self, dataset):
        live = getattr(self.camera, "model", None)
        if live is None:
            return
        saved = dataset.camera
        if (live.width, live.height, live.distortion_model) != (saved.width, saved.height, saved.distortion_model):
            raise HandeyeError("RESUME_CAMERA", "当前相机分辨率或畸变模型与会话不一致，请新建会话")
        if not np.allclose(live.K, saved.K, atol=0.5) or not np.allclose(np.asarray(live.D), np.asarray(saved.D), atol=1e-4):
            raise HandeyeError("RESUME_CAMERA", "当前相机内参与会话差异过大，请新建会话")

    def resume_session(self, path):
        with self.lock:
            self._editable()
            dataset = Dataset(path)
            saved = dataset.session["config"]
            if saved["frames"] != self.config["frames"]:
                raise HandeyeError("RESUME_MISMATCH", "会话坐标系与当前运行配置不一致")
            if saved["robot"]["pose_source"] != self.config["robot"]["pose_source"]:
                raise HandeyeError("RESUME_MISMATCH", "会话位姿来源与当前运行配置不一致")
            self._camera_compatible(dataset)
            self.dataset = dataset
            self.config = deepcopy(saved)
            remember_session(self.config["storage"]["root"], dataset.path)
        report = self.recover_pending()
        with self.lock:
            self._load_results()
        return report

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

