from threading import Condition, Event, Thread
import time
from importlib.metadata import version, PackageNotFoundError
import cv2
import numpy as np
from .errors import HandeyeError
from .models import CameraModel, Frame


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
            v = profile.as_video_stream_profile()
            intr = v.get_intrinsics()
            distortion = str(intr.model).split(".")[-1]
            try:
                sdk_version = version("pyrealsense2")
            except PackageNotFoundError:
                sdk_version = getattr(rs, "__version__", "unknown")
            self.model = CameraModel(intr.width, intr.height,
                                     [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                                     list(intr.coeffs), distortion,
                                     {**selected, "serial": device["serial"], "firmware": device["firmware"],
                                      "device_name": device["name"], "sdk_version": sdk_version,
                                      "fallback_used": selected != {"stream": cfg["stream"], "index": cfg["stream_index"],
                                          "format": cfg["format"], "width": cfg["width"], "height": cfg["height"], "fps": cfg["fps"]}})
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
        try:
            while not self._stop.is_set():
                frames = self.pipeline.wait_for_frames(1000)
                frame = frames.first_or_default(getattr(rs.stream, self.model.profile["stream"]),
                                                 self.model.profile["index"])
                if not frame:
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

