from copy import deepcopy
from pathlib import Path
import hashlib
import json
import yaml
from .errors import HandeyeError

DEFAULTS = {
    "calibration": {"type": "eye_in_hand", "algorithms": ["PARK", "TSAI"],
                    "recommended_training_samples": 25},
    "frames": {"base": "base_link", "end": "l_end_link",
               "camera_optical": "d405_color_optical_frame"},
    "camera": {"backend": "realsense", "serial": "", "stream": "color", "stream_index": 0,
               "width": 848, "height": 480, "fps": 30, "format": "bgr8",
               "allow_profile_fallback": False, "enable_global_time_if_supported": True},
    "robot": {"pose_source": "service_request", "serial": "", "service_name": "/robot/get_end_pose",
              "query_timeout_seconds": 1.5, "allow_inverted_input": False,
              "compat_topic": "/robot/end_pose",
              "compat_message_type": "geometry_msgs/msg/TransformStamped",
              "compat_subscription_timeout_seconds": 3.0, "max_source_age_seconds": 2.0},
    "sampling": {"pairing_mode": "manual_stationary_snapshot", "pose_reads_per_capture": 1,
                 "capture_timeout_seconds": 5.0},
    "auto_motion": {"board_center_m": [0.0, 1.32, 0.0], "points": 60, "max_tilt_deg": 20.0,
                    "max_roll_deg": 20.0, "max_travel_m": 0.15, "move_group": "left_arm",
                    "velocity_scale": 0.1, "joint_limit_margin_rad": 0.1},
    "board": {"count_interpretation": "inner_corners", "cols": 9, "rows": 6,
              "square_length_m": 0.043,
              "physical_origin": "固定同一物理角为 0，沿长边为 X、短边为 Y；人工核对编号"},
    "quality": {"warning_pnp_rmse_px": 0.8, "warning_hull_area_ratio": 0.03},
    "validation": {"warning_position_rmse_mm": 2.0, "warning_rotation_rmse_deg": 0.5},
    "storage": {"root": "./handeye_data"},
    "ui": {"preview_fps": 10},
}


def _merge(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value


def load_config(path=None):
    cfg = deepcopy(DEFAULTS)
    if path:
        path = Path(path).resolve()
        with path.open(encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
        cfg["_config_path"] = str(path)
        root = Path(cfg["storage"]["root"])
        cfg["storage"]["root"] = str(root if root.is_absolute() else path.parent.parent / root)
    if cfg["calibration"]["type"] != "eye_in_hand":
        raise HandeyeError("CONFIG", "当前版本只实现眼在手上")
    if cfg["sampling"]["pose_reads_per_capture"] != 1:
        raise HandeyeError("CONFIG", "每次拍照固定读取一次；不实现两次读取或稳定检查")
    if cfg["sampling"]["pairing_mode"] != "manual_stationary_snapshot":
        raise HandeyeError("CONFIG", "当前使用人工保持不动的单次快照，不提供硬同步模式")
    for name in ("query_timeout_seconds", "compat_subscription_timeout_seconds", "max_source_age_seconds"):
        if cfg["robot"][name] <= 0:
            raise HandeyeError("CONFIG", f"{name} 必须大于零")
    if cfg["sampling"]["capture_timeout_seconds"] <= 0 or cfg["ui"]["preview_fps"] <= 0:
        raise HandeyeError("CONFIG", "采集超时和预览频率必须大于零")
    auto = cfg["auto_motion"]
    if (len(auto["board_center_m"]) != 3 or auto["points"] < 1 or not 0 < auto["max_tilt_deg"] < 90
            or auto["max_travel_m"] <= 0 or not 0 < auto["velocity_scale"] <= 1
            or auto["joint_limit_margin_rad"] < 0):
        raise HandeyeError("CONFIG", "auto_motion 需要三维 board_center_m、至少 1 个点、0 到 90 度之间的 max_tilt_deg、"
                                     "正的 max_travel_m，velocity_scale 在 0 到 1 之间，joint_limit_margin_rad 不能为负")
    if cfg["robot"]["pose_source"] == "flexiv_sdk" and not str(cfg["robot"].get("serial", "")).strip():
        raise HandeyeError("CONFIG", "Flexiv 位姿源需要 robot.serial")
    return cfg


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()

