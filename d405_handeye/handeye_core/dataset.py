from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import csv
import json
import os
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
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise HandeyeError("IMAGE_WRITE", "原图 PNG 编码失败")
            with (temp / "image.png").open("wb") as f:
                f.write(encoded.tobytes())
                f.flush()
                os.fsync(f.fileno())
            record = deepcopy(record)
            record.update(sample_id=sample_id, geometry_fingerprint=self.session["geometry_fingerprint"])
            write_json(temp / "sample.json", record)
            os.replace(temp, folder)
            index = deepcopy(self.index)
            index["samples"].append({"sample_id": sample_id, "role": record["role"], "enabled": True,
                                     "label": record.get("label", ""), "analysis_path": None})
            self._commit(index)
        return sample_id

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
            writer = csv.DictWriter(f, fieldnames=["sample_id", "role", "position_error_mm", "rotation_error_deg",
                                                  "pnp_rmse_px", "predicted_reprojection_rmse_px"])
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

