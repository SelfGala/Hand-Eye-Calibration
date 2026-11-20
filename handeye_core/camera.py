from threading import Condition, Event, Thread
import time
from importlib.metadata import version, PackageNotFoundError
import cv2
import numpy as np
from .errors import HandeyeError
from .models import CameraModel, Frame


def _has_configured_intrinsics(cfg):
    has_k, has_d = "K" in cfg, "D" in cfg
    if has_k != has_d:
        raise HandeyeError("CONFIG", "K 和 D 需要同时写在配置里，或者都不写并使用相机读到的内参")
    return has_k and has_d


def _distortion_name(model):
    return str(model).split(".")[-1].strip().lower()


def camera_model_from_realsense(intrinsics, profile):
    """Copy the opened stream's intrinsics. Coefficients are not rewritten into another model."""
    supported = {
        "none": "none",
        "brown_conrady": "brown_conrady",
        "inverse_brown_conrady": "inverse_brown_conrady",
    }
    name = _distortion_name(intrinsics.model)
    if name not in supported:
        raise HandeyeError("CAMERA_MODEL_UNSUPPORTED",
                           f"相机报告的畸变模型是 {name}，只按原样接受 none、brown_conrady、inverse_brown_conrady")
    distortion_model = supported[name]
    coeffs = [float(value) for value in list(intrinsics.coeffs)]
    if distortion_model == "none":
        coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    elif len(coeffs) < 5:
        raise HandeyeError("CAMERA_MODEL", "相机内参畸变系数不足 5 个")
    else:
        coeffs = coeffs[:5]
    matrix = [[float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
              [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
              [0.0, 0.0, 1.0]]
    return CameraModel(int(intrinsics.width), int(intrinsics.height), matrix, coeffs, distortion_model,
                       {**profile, "intrinsics_source": "device"})


def camera_model_from_config(cfg, profile):
    """Build the active camera model from an explicit config file."""
    missing = [key for key in ("K", "D", "distortion_model") if key not in cfg]
    if missing:
        raise HandeyeError("CONFIG", "相机配置缺少 " + "、".join(missing) + "。要使用这份配置内参时，K、D 和畸变模型都要写上")
    matrix = np.asarray(cfg["K"], dtype=float)
    if matrix.size != 9:
        raise HandeyeError("CONFIG", "K 必须是 3×3 或 9 个数")
    distortion = np.asarray(cfg["D"], dtype=float).reshape(-1)
    return CameraModel(cfg["width"], cfg["height"], matrix.reshape(3, 3).tolist(),
                       distortion.tolist(), str(cfg["distortion_model"]), profile)


def realsense_devices():
    import pyrealsense2 as rs
    devices = []
    for device in rs.context().query_devices():
        info = {"name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version), "sensors": []}
        for sensor in device.query_sensors():
            profiles = []
            for p in sensor.get_stream_profiles():
                if p.is_video_stream_profile():
                    v = p.as_video_stream_profile()
                    profiles.append({"stream": str(p.stream_type()).split(".")[-1], "index": p.stream_index(),
                                     "width": v.width(), "height": v.height(), "format": str(p.format()).split(".")[-1],
                                     "fps": p.fps()})
            info["sensors"].append({"name": sensor.get_info(rs.camera_info.name), "profiles": profiles})
        devices.append(info)
    return devices


class RealSenseCamera:
    def __init__(self, config):
        self.config = config
        self._condition = Condition()
        self._stop = Event()
        self._latest = None
        self.error = None
        self._thread = None
        self.pipeline = None
        self.model = None

    def start(self):
        import pyrealsense2 as rs
        cfg = self.config
        if cfg["format"] not in ("bgr8", "rgb8", "y8"):
            raise HandeyeError("CAMERA_FORMAT", "支持 bgr8、rgb8 或 y8 图像格式")
        use_config_intrinsics = _has_configured_intrinsics(cfg)
        devices = realsense_devices()
        if not devices:
            raise HandeyeError("CAMERA_NOT_FOUND", "未发现 RealSense 相机")
        if cfg["serial"]:
            device = next((d for d in devices if d["serial"] == cfg["serial"]), None)
            if device is None:
                raise HandeyeError("CAMERA_NOT_FOUND", f"未找到序列号 {cfg['serial']}")
        elif len(devices) == 1:
            device = devices[0]
        else:
            raise HandeyeError("CAMERA_AMBIGUOUS", "存在多台相机，请配置 serial")
        candidates = [p for s in device["sensors"] for p in s["profiles"]
                      if p["stream"] == cfg["stream"] and p["index"] == cfg["stream_index"]
                      and p["format"] == cfg["format"]]
        selected = {"stream": cfg["stream"], "index": cfg["stream_index"], "format": cfg["format"],
                    "width": cfg["width"], "height": cfg["height"], "fps": cfg["fps"]}
        if selected not in candidates:
            if not cfg["allow_profile_fallback"] or not candidates:
                raise HandeyeError("PROFILE_UNSUPPORTED", f"配置不支持；可用候选：{candidates}")
            selected = min(candidates, key=lambda p: (abs(p["width"] - cfg["width"]) +
                                                     abs(p["height"] - cfg["height"]), abs(p["fps"] - cfg["fps"])))
        pipeline = rs.pipeline()
        request = rs.config()
        request.enable_device(device["serial"])
        request.enable_stream(getattr(rs.stream, selected["stream"]), selected["index"],
                              selected["width"], selected["height"], getattr(rs.format, selected["format"]),
                              selected["fps"])
        request.resolve(rs.pipeline_wrapper(pipeline))
        active = pipeline.start(request)
        try:
            if cfg["enable_global_time_if_supported"]:
                for sensor in active.get_device().query_sensors():
                    if sensor.supports(rs.option.global_time_enabled):
                        sensor.set_option(rs.option.global_time_enabled, 1)
            profile = active.get_stream(getattr(rs.stream, selected["stream"]), selected["index"])
            opened = profile.as_video_stream_profile()
            try:
                sdk_version = version("pyrealsense2")
            except PackageNotFoundError:
                sdk_version = getattr(rs, "__version__", "unknown")
            fallback_used = selected != {"stream": cfg["stream"], "index": cfg["stream_index"],
                                         "format": cfg["format"], "width": cfg["width"], "height": cfg["height"], "fps": cfg["fps"]}
            profile_meta = {**selected, "serial": device["serial"], "firmware": device["firmware"],
                            "device_name": device["name"], "sdk_version": sdk_version,
                            "fallback_used": fallback_used, "width": opened.width(), "height": opened.height()}
            if use_config_intrinsics:
                if (opened.width(), opened.height()) != (cfg["width"], cfg["height"]):
                    raise HandeyeError("PROFILE_UNSUPPORTED",
                                       f"图像是 {opened.width()}×{opened.height()}，配置内参对应 {cfg['width']}×{cfg['height']}")
                self.model = camera_model_from_config(cfg, {**profile_meta, "intrinsics_source": "config"})
            else:
                intrinsics = opened.get_intrinsics()
                if (intrinsics.width, intrinsics.height) != (opened.width(), opened.height()):
                    raise HandeyeError("CAMERA_MODEL", "相机内参分辨率和实际图像不一致")
                self.model = camera_model_from_realsense(intrinsics, profile_meta)
        except BaseException:
            pipeline.stop()
            raise
        self.pipeline = pipeline
        self._stop.clear()
        self._thread = Thread(target=self._loop, name="realsense-camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        import pyrealsense2 as rs
        stream = getattr(rs.stream, self.model.profile["stream"])
        index = self.model.profile["index"]
        try:
            while not self._stop.is_set():
                frames = self.pipeline.wait_for_frames(1000)
                frame = None
                for item in frames:
                    profile = item.get_profile()
                    if profile.stream_type() == stream and profile.stream_index() == index:
                        frame = item
                        break
                if frame is None:
                    continue
                image = np.asanyarray(frame.get_data()).copy()
                if self.model.profile["format"] == "rgb8":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                packet = Frame(image, frame.get_frame_number(), frame.get_timestamp(),
                               str(frame.get_frame_timestamp_domain()))
                with self._condition:
                    self._latest = packet
                    self._condition.notify_all()
        except Exception as exc:
            if not self._stop.is_set():
                with self._condition:
                    self.error = str(exc)
                    self._condition.notify_all()

    def latest(self):
        with self._condition:
            return self._latest

    def fresh(self, after_monotonic_ns, timeout, cancel):
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if cancel.is_set():
                    raise HandeyeError("CANCELLED", "采样已取消")
                if self.error:
                    raise HandeyeError("CAMERA_READ", self.error)
                if self._latest is not None and self._latest.host_monotonic_ns > after_monotonic_ns:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HandeyeError("CAMERA_TIMEOUT", "未收到新图像")
                self._condition.wait(min(remaining, 0.05))

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None

