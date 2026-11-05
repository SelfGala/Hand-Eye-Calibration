import importlib
import importlib.util
import os
from pathlib import Path
import platform
import sys


def diagnose(config, camera=False, ros_graph=False, ros_args=None):
    result = {"platform": platform.platform(), "architecture": platform.machine(),
              "python": sys.version, "executable": sys.executable,
              "ROS_DISTRO": os.environ.get("ROS_DISTRO"), "RMW_IMPLEMENTATION": os.environ.get("RMW_IMPLEMENTATION"),
              "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID"), "DISPLAY": os.environ.get("DISPLAY"),
              "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"), "robot_queries": 0,
              "modules": {}}
    for name in ("numpy", "scipy", "cv2", "yaml", "PyQt5.QtCore", "pyrealsense2", "rclpy"):
        try:
            module = importlib.import_module(name)
            result["modules"][name] = {"available": True, "version": getattr(module, "__version__",
                getattr(module, "PYQT_VERSION_STR", "unknown")), "path": getattr(module, "__file__", None)}
        except Exception as exc:
            result["modules"][name] = {"available": False, "error": str(exc)}
    result["same_interpreter_ros_and_sdk"] = all(result["modules"][n]["available"] for n in ("rclpy", "pyrealsense2"))
    l4t = Path("/etc/nv_tegra_release")
    result["l4t"] = l4t.read_text().strip() if l4t.exists() else None
    if camera:
        try:
            from .camera import realsense_devices, RealSenseCamera
            result["devices"] = realsense_devices()
            stream = RealSenseCamera(config["camera"])
            try:
                stream.start()
                result["active_camera"] = stream.model.to_dict()
            finally:
                stream.close()
        except Exception as exc:
            result["camera_error"] = str(exc)
    if ros_graph:
        try:
            from .ros_runtime import RosRuntime
            runtime = RosRuntime(config, ros_args)
            try:
                result["pose_service_available"] = runtime.pose_source.service_available(2.0)
            finally:
                runtime.close()
        except Exception as exc:
            result["ros_graph_error"] = str(exc)
    return result

