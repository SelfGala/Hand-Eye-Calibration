from dataclasses import dataclass, asdict, field
import time
import numpy as np
from .errors import HandeyeError


@dataclass(frozen=True)
class Board:
    count_interpretation: str = "inner_corners"
    cols: int = 9
    rows: int = 6
    square_length_m: float = 0.043
    physical_origin: str = "固定同一物理角为 0，X/Y 沿棋盘行列方向"

    def __post_init__(self):
        if self.count_interpretation not in ("inner_corners", "squares"):
            raise HandeyeError("BOARD_CONFIG", "棋盘计数含义应为 inner_corners 或 squares")
        if min(self.pattern_size) < 2 or self.cols == self.rows:
            raise HandeyeError("BOARD_CONFIG", "需要至少 2×2 的非正方形内角点网格")
        if not np.isfinite(self.square_length_m) or self.square_length_m <= 0:
            raise HandeyeError("BOARD_CONFIG", "单格尺寸必须为正数（米）")

    @property
    def pattern_size(self):
        d = int(self.count_interpretation == "squares")
        return self.cols - d, self.rows - d

    def object_points(self):
        cols, rows = self.pattern_size
        points = np.zeros((cols * rows, 3), dtype=np.float64)
        points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        return points * self.square_length_m

    def to_dict(self):
        return asdict(self)


@dataclass
class CameraModel:
    width: int
    height: int
    K: list
    D: list
    distortion_model: str = "none"
    profile: dict = field(default_factory=dict)

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=float).reshape(3, 3).tolist()
        self.D = np.asarray(self.D, dtype=float).reshape(-1).tolist()
        K = np.array(self.K)
        if (self.width <= 0 or self.height <= 0 or not np.isfinite(K).all()
                or not np.isfinite(self.D).all() or min(K[0, 0], K[1, 1]) <= 0
                or not np.allclose(K[2], [0, 0, 1])):
            raise HandeyeError("CAMERA_MODEL", "内参尺寸、焦距或数值无效")
        if self.distortion_model not in ("none", "brown_conrady", "opencv", "inverse_brown_conrady"):
            raise HandeyeError("CAMERA_MODEL_UNSUPPORTED",
                               f"尚未适配畸变模型 {self.distortion_model}，不能当作 OpenCV 系数")
        if self.distortion_model == "none" and np.any(np.array(self.D) != 0):
            raise HandeyeError("CAMERA_MODEL", "NONE 模型包含非零畸变系数")
        if self.D and len(self.D) not in (4, 5, 8, 12, 14):
            raise HandeyeError("CAMERA_MODEL", "OpenCV 畸变系数数量无效")
        if self.distortion_model == "inverse_brown_conrady" and len(self.D) != 5:
            raise HandeyeError("CAMERA_MODEL", "RealSense 反向 Brown-Conrady 需要 5 个系数")

    def to_dict(self):
        return asdict(self)


@dataclass
class Frame:
    image: np.ndarray
    number: int
    sdk_timestamp_ms: float
    timestamp_domain: str
    host_wall_ns: int = field(default_factory=time.time_ns)
    host_monotonic_ns: int = field(default_factory=time.monotonic_ns)

    def metadata(self):
        return {k: v for k, v in asdict_without_image(self).items()}


def asdict_without_image(frame):
    return {"frame_number": frame.number, "sdk_timestamp_ms": frame.sdk_timestamp_ms,
            "timestamp_domain": frame.timestamp_domain, "host_wall_ns": frame.host_wall_ns,
            "host_monotonic_ns": frame.host_monotonic_ns}


@dataclass
class PoseReading:
    G: np.ndarray
    metadata: dict

