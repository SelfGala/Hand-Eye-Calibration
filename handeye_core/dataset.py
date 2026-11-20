from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import csv
import json
import os
import shutil
import uuid
import cv2
import numpy as np
import yaml
from .config import fingerprint
from .errors import HandeyeError
from .models import Board, CameraModel


def new_id(prefix):
    return f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:10]}"


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HandeyeError("IMAGE_READ", f"无法读取原图 {path}")
    return image


def write_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise HandeyeError("IMAGE_WRITE", "原图 PNG 编码失败")
    with Path(path).open("wb") as f:
        f.write(encoded.tobytes())
        f.flush()
        os.fsync(f.fileno())


def _history_path(root):
    return Path(root).resolve() / "latest_session.json"


def _read_history(root):
    path = _history_path(root)
    if not path.is_file():
        return {"schema_version": 2, "latest": "", "history": []}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("history", [])
    return data


def remember_session(root, session_path):
    resolved = str(Path(session_path).resolve())
    data = _read_history(root)
    history = [resolved] + [item for item in data.get("history", []) if item != resolved]
    write_json(_history_path(root), {"schema_version": 2, "latest": resolved, "history": history[:20]})


def _resumable(path):
    path = Path(path)
    index_path = path / "index.json"
    if not (path / "session.yaml").is_file() or not index_path.is_file():
        return False
    with index_path.open(encoding="utf-8") as f:
        index = json.load(f)
    if index.get("samples"):
        return True
    pending = path / "cache" / "pending"
    return pending.is_dir() and any(item.is_dir() and not item.name.startswith(".") for item in pending.iterdir())


def session_candidates(root):
    """History first, then any session directory under the storage root."""
    root = Path(root).resolve()
    ordered, seen = [], set()
    def add(item):
        try:
            resolved = str(Path(item).resolve())
        except OSError:
            return
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    for item in _read_history(root).get("history", []):
        add(item)
    if root.is_dir():
        children = [path for path in root.iterdir() if path.is_dir() and (path / "session.yaml").is_file()]
        children.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for child in children:
            add(child)
    return ordered


def latest_resumable(root):
    for item in session_candidates(root):
        if _resumable(item):
            return Path(item)
    raise HandeyeError("NO_SESSION", "没有可继续的标定会话")


def previous_session(root, current):
    current = str(Path(current).resolve())
    for item in session_candidates(root):
        if item != current and _resumable(item):
            return Path(item)
    raise HandeyeError("NO_SESSION", "没有更早的标定会话可继续")


def resolve_resume(value, root):
    if value and value != "latest":
        path = Path(value).resolve()
        if not (path / "session.yaml").is_file():
            raise HandeyeError("NO_SESSION", f"会话不存在：{path}")
        return path
    return latest_resumable(root)


class Dataset:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.lock = RLock()
        with (self.path / "session.yaml").open(encoding="utf-8") as f:
            self.session = yaml.safe_load(f)
        if self.session.get("schema_version") != 2:
            raise HandeyeError("DATASET_VERSION", "仅支持 schema_version=2，不自动解释旧数据")
        with (self.path / "index.json").open(encoding="utf-8") as f:
            self.index = json.load(f)
        self.board = Board(**self.session["board"])
        self.camera = CameraModel(**self.session["camera"])

    @classmethod
    def create(cls, root, config, camera, environment=None):
        path = Path(root).resolve() / new_id("session")
        path.mkdir(parents=True)
        board = Board(**config["board"])
        geometry = {"board": board.to_dict(), "camera": camera.to_dict(), "frames": config["frames"]}
        session = {"schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
                   "config": deepcopy(config), **geometry, "geometry_fingerprint": fingerprint(geometry),
                   "robot_read_policy": "capture_only", "environment": environment or {}}
        atomic_write(path / "session.yaml", yaml.safe_dump(session, allow_unicode=True, sort_keys=False))
        atomic_write(path / "camera.yaml", yaml.safe_dump(camera.to_dict(), sort_keys=False))
        atomic_write(path / "board.yaml", yaml.safe_dump(board.to_dict(), allow_unicode=True))
        write_json(path / "index.json", {"schema_version": 2, "dataset_revision": 0, "samples": []})
        remember_session(path.parent, path)
        return cls(path)

    @property
    def revision(self):
        with self.lock:
            return self.index["dataset_revision"]

    def _commit(self, index):
        index["dataset_revision"] += 1
        write_json(self.path / "revisions" / f"index_{index['dataset_revision']:06d}.json", index)
        write_json(self.path / "index.json", index)
        self.index = index

    def add(self, image, record):
        if record["role"] not in ("train", "validation"):
            raise HandeyeError("ROLE", "角色必须为 train 或 validation")
        sample_id = new_id("sample")
        with self.lock:
            folder = self.path / "samples" / sample_id
            temp = folder.with_name(".pending_" + sample_id)
            temp.mkdir(parents=True)
            write_png(temp / "image.png", image)
            record = deepcopy(record)
            record.update(sample_id=sample_id, geometry_fingerprint=self.session["geometry_fingerprint"])
            write_json(temp / "sample.json", record)
            os.replace(temp, folder)
            index = deepcopy(self.index)
            index["samples"].append({"sample_id": sample_id, "role": record["role"], "enabled": True,
                                     "label": record.get("label", ""), "analysis_path": None})
            self._commit(index)
        return sample_id

    def stage_capture(self, image, payload):
        """Write image, arm pose and TF before detection so a crash can resume."""
        capture_id = new_id("capture")
        record = deepcopy(payload)
        record.update(schema_version=2, capture_id=capture_id, state="pending", error_code="", error="")
        with self.lock:
            folder = self.path / "cache" / "pending" / capture_id
            temp = folder.with_name(".pending_" + capture_id)
            temp.mkdir(parents=True)
            write_png(temp / "image.png", image)
            write_json(temp / "capture.json", record)
            os.replace(temp, folder)
        return capture_id

    def pending_captures(self):
        root = self.path / "cache" / "pending"
        if not root.is_dir():
            return []
        records = []
        with self.lock:
            folders = sorted(item for item in root.iterdir() if item.is_dir() and not item.name.startswith("."))
            for folder in folders:
                with (folder / "capture.json").open(encoding="utf-8") as f:
                    record = json.load(f)
                record["image_path"] = str(folder / "image.png")
                records.append(record)
        return records

    def _cache_record_path(self, capture_id):
        folder = self.path / "cache" / "pending" / capture_id
        if not (folder / "capture.json").is_file():
            raise HandeyeError("CACHE_NOT_FOUND", capture_id)
        return folder

    def mark_cache_error(self, capture_id, code, message):
        with self.lock:
            folder = self._cache_record_path(capture_id)
            with (folder / "capture.json").open(encoding="utf-8") as f:
                record = json.load(f)
            record.update(state="failed", error_code=code, error=message)
            write_json(folder / "capture.json", record)

    def discard_cache(self, capture_id):
        with self.lock:
            folder = self.path / "cache" / "pending" / capture_id
            if folder.is_dir():
                shutil.rmtree(folder)

    def finalize_cache(self, capture_id):
        self.discard_cache(capture_id)

    def find_sample_by_request(self, request_id):
        if not request_id:
            return None
        with self.lock:
            for entry in self.index["samples"]:
                path = self.path / "samples" / entry["sample_id"] / "sample.json"
                if not path.is_file():
                    continue
                with path.open(encoding="utf-8") as f:
                    record = json.load(f)
                if record.get("transaction", {}).get("request_id") == request_id:
                    return entry["sample_id"]
        return None

    def snapshot(self):
        with self.lock:
            samples = []
            for entry in self.index["samples"]:
                folder = self.path / "samples" / entry["sample_id"]
                with (folder / "sample.json").open(encoding="utf-8") as f:
                    record = json.load(f)
                if entry["analysis_path"]:
                    with (self.path / entry["analysis_path"]).open(encoding="utf-8") as f:
                        record["analysis"] = json.load(f)["analysis"]
                record.update(deepcopy(entry))
                samples.append(record)
            return self.revision, samples

    def update(self, sample_id, *, enabled=None, role=None, label=None):
        with self.lock:
            index = deepcopy(self.index)
            entry = next((e for e in index["samples"] if e["sample_id"] == sample_id), None)
            if entry is None:
                raise HandeyeError("SAMPLE_NOT_FOUND", sample_id)
            if role is not None:
                if role not in ("train", "validation"):
                    raise HandeyeError("ROLE", "角色必须为 train 或 validation")
                entry["role"] = role
            if enabled is not None:
                entry["enabled"] = bool(enabled)
            if label is not None:
                entry["label"] = label
            self._commit(index)

    def replace_analyses(self, replacements, reason):
        """Write all analyses, then commit a single index revision; never change raw measurements."""
        with self.lock:
            index = deepcopy(self.index)
            for sample_id, analysis in replacements.items():
                entry = next(e for e in index["samples"] if e["sample_id"] == sample_id)
                rel = Path("analyses") / sample_id / (new_id("analysis") + ".json")
                write_json(self.path / rel, {"reason": reason, "analysis": analysis,
                                             "based_on_revision": self.revision})
                entry["analysis_path"] = rel.as_posix()
            self._commit(index)

    def save_result(self, result, revision):
        result = deepcopy(result)
        result_id = new_id(result["method"].lower())
        result.update(result_id=result_id, dataset_revision=revision,
                      parent_frame=self.session["frames"]["end"],
                      child_frame=self.session["frames"]["camera_optical"],
                      geometry_fingerprint=self.session["geometry_fingerprint"],
                      board=self.session["board"], camera=self.session["camera"],
                      environment=self.session.get("environment", {}))
        folder = self.path / "results" / result_id
        temp = folder.with_name(".pending_" + result_id)
        temp.mkdir(parents=True)
        write_json(temp / "report.json", result)
        atomic_write(temp / "calibration.yaml", yaml.safe_dump(result, allow_unicode=True, sort_keys=False))
        rows = result["metrics"]["training"]["per_sample"] + result["metrics"]["validation"]["per_sample"]
        with (temp / "report.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "role", "position_error_mm",
                                                  "position_error_x_mm", "position_error_y_mm", "position_error_z_mm",
                                                  "rotation_error_deg", "pnp_rmse_px", "predicted_reprojection_rmse_px"])
            writer.writeheader()
            writer.writerows(rows)
        text = (f"{result_id}\n算法 {result['method']}；数据版本 {revision}\n"
                f"{result['parent_frame']} <- {result['child_frame']}，米 / xyzw\n"
                f"验证状态：{'有独立验证' if result['has_validation_metrics'] else '未验证'}\n"
                "手眼链像素误差由训练集板位姿预测，验证样本不参与拟合。\n"
                "采样依赖人工保持不动；没有稳定等待或运动状态门槛。\n"
                + "\n".join(result["warnings"]) + "\n\n" + json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        atomic_write(temp / "report.txt", text)
        os.replace(temp, folder)
        return result

