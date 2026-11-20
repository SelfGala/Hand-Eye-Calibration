"""Deterministic pinhole rendering and a clearly labelled, hardware-free demo."""
from threading import Event
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .dataset import Dataset
from .detection import project
from .models import Board, CameraModel, Frame, PoseReading
from .transforms import transform, inverse
from .errors import HandeyeError


def scene():
    X = transform(Rotation.from_euler("xyz", [0.12, -0.08, 0.18]).as_matrix(), [0.06, -0.025, 0.09])
    Y = transform(Rotation.from_euler("xyz", [0.1, -0.2, 0.05]).as_matrix(), [0.5, 0.1, 0.25])
    camera = CameraModel(848, 480, [[620, 0, 424], [0, 620, 240], [0, 0, 1]], [0] * 5,
                         profile={"source": "SYNTHETIC_DEMO", "stream": "color", "index": 0,
                                  "format": "bgr8", "fps": 30, "serial": "DEMO"})
    return X, Y, camera


def pose_at(index, board):
    rng = np.random.default_rng(1301 + index)
    angles = rng.uniform([-0.38, -0.38, -0.35], [0.38, 0.38, 0.35])
    cols, rows = board.pattern_size
    center = np.array([(cols - 1) * board.square_length_m / 2,
                       (rows - 1) * board.square_length_m / 2, 0])
    R = Rotation.from_euler("xyz", angles).as_matrix()
    target = np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.025, 0.025), rng.uniform(0.95, 1.2)])
    return transform(R, target - R @ center)


def render(board, camera, C):
    image = np.full((camera.height, camera.width, 3), 210, np.uint8)
    cols, rows = board.pattern_size
    for y in range(-1, rows):
        for x in range(-1, cols):
            points = np.array([[x, y, 0], [x + 1, y, 0], [x + 1, y + 1, 0], [x, y + 1, 0]], float)
            polygon = np.round(project(points * board.square_length_m, C, camera)).astype(np.int32)
            color = 20 if (x + y) % 2 == 0 else 245
            cv2.fillConvexPoly(image, polygon, (color,) * 3, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(image, (3, 3), 0.4)


class DemoRig:
    def __init__(self, board):
        self.board = board
        self.X, self.Y, self.model = scene()
        self.index = 0
        self.frame_number = 0
        self.query_count = 0
        self.closed = False
        self._cached_image = None
        self._image_index = -1

    def read(self, request_id, timeout, cancel, on_request):
        if cancel.is_set():
            raise HandeyeError("CANCELLED", "已取消")
        on_request()
        self.query_count += 1
        self.index += 1
        C = pose_at(self.index, self.board)
        G = self.Y @ inverse(C) @ inverse(self.X)
        return PoseReading(G, {"request_id": request_id, "source_kind": "replay",
                                "timestamp_kind": "synthetic", "sdk_call_count": 0,
                                "source_age_known": False, "has_motion_state": False,
                                "demo": True})

    def latest(self):
        if self._image_index != self.index:
            self._cached_image = render(self.board, self.model, pose_at(self.index, self.board))
            self._image_index = self.index
        self.frame_number += 1
        return Frame(self._cached_image.copy(), self.frame_number, time.monotonic() * 1000, "synthetic")

    def fresh(self, after_monotonic_ns, timeout, cancel):
        if cancel.is_set():
            raise HandeyeError("CANCELLED", "已取消")
        return self.latest()

    def close(self):
        self.closed = True


def create_dataset(root, config, train_count=20, validation_count=5, noise_px=0.0):
    from .detection import from_corners
    board = Board(**config["board"])
    X, Y, camera = scene()
    dataset = Dataset.create(root, config, camera, {"kind": "synthetic", "seed": 314159})
    rng = np.random.default_rng(314159)
    for i in range(train_count + validation_count):
        C = pose_at(i, board)
        G = Y @ inverse(C) @ inverse(X)
        pixels = project(board.object_points(), C, camera) + rng.normal(0, noise_px, (len(board.object_points()), 2))
        analysis = from_corners(pixels, board, camera)
        dataset.add(render(board, camera, C), {"schema_version": 2,
            "role": "train" if i < train_count else "validation", "G": G.tolist(), "analysis": analysis,
            "robot": {"source_kind": "replay", "timestamp_kind": "synthetic", "sdk_call_count": 0},
            "camera_frame": {"frame_number": i, "timestamp_domain": "synthetic"},
            "transaction": {"pose_request_count": 0, "stability_check": "unavailable"},
            "label": f"SYNTHETIC_{i}", "warnings": ["合成数据，不能用于真实机械臂"]})
    return dataset, X

