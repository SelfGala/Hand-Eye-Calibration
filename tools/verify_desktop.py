"""Finite native-UI integration check. No hardware, no idle/stability waits."""
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)  # Deterministic CPU QA; no host GPU cache dependency.
from PyQt5.QtWidgets import QApplication
from handeye_core.config import load_config
from handeye_core.controller import Controller
from handeye_core.dataset import Dataset, write_json
from handeye_core.detection import project
from handeye_core.models import Board
from handeye_core.synthetic import DemoRig, pose_at
from handeye_gui.main_window import MainWindow
from handeye_gui.fonts import configure_fonts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config()
    config["storage"]["root"] = str(Path(args.work).resolve())
    rig = DemoRig(Board(**config["board"]))
    dataset = Dataset.create(config["storage"]["root"], config, rig.model)
    controller = Controller(config, dataset, rig, rig)
    app = QApplication.instance() or QApplication(["verify-native-handeye"])
    configure_fonts(app)
    window = MainWindow(controller, True)
    window.show()
    jobs = []
    corrections = 0
    def await_current():
        job_id = window.current_job
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            app.processEvents()
            job = controller.get_job(job_id)
            if job["state"] in ("succeeded", "failed", "cancelled"):
                app.processEvents()
                window.tick()
                if job["state"] != "succeeded":
                    raise AssertionError(job)
                jobs.append(job)
                return job
            time.sleep(0.005)
        raise AssertionError("native workflow timeout")
    try:
        # Finite preview calls demonstrate that the GUI does not request robot state.
        for _ in range(5):
            window.tick()
            app.processEvents()
        assert rig.query_count == 0
        deadline = time.monotonic() + 5
        while window.displayed_image is None and time.monotonic() < deadline:
            app.processEvents()
            window.tick()
            time.sleep(0.005)
        assert window.displayed_image is not None
        app.processEvents()
        assert window.grab().save(str(output / "gui-preview.png"))
        for i in range(20):
            window.role.setCurrentIndex(0 if i < 15 else 1)
            window.capture()
            job = await_current()
            assert job["pose_request_count"] == 1
            _, samples = dataset.snapshot()
            sample = samples[-1]
            # A test renderer has known physical corner identities. Use that truth only
            # in this verification harness to exercise the same manual correction button.
            expected = project(dataset.board.object_points(), pose_at(rig.index, dataset.board), rig.model)
            actual = np.asarray(sample["analysis"]["corners"])
            if np.linalg.norm(actual[0] - expected[0]) > np.linalg.norm(actual[-1] - expected[0]):
                window.table.selectRow(len(samples) - 1)
                window.flip_sample()
                corrections += 1
        assert rig.query_count == 20
        window.table.selectRow(0)
        window.view_sample(0, 0)
        app.processEvents()
        assert window.grab().save(str(output / "gui-samples.png"))
        window.compute()
        job = await_current()
        assert job["pose_request_count"] == 0 and rig.query_count == 20
        assert len(job["result_ids"]) == 2
        window.tabs.setCurrentIndex(1)
        app.processEvents()
        assert window.grab().save(str(output / "gui-results.png"))
        result = controller.get_result(window.result_combo.currentData())
        report = {"native_qt": True, "preview_pose_requests": 0, "capture_count": 20,
                  "pose_requests": rig.query_count, "compute_pose_requests": 0,
                  "manual_orientation_corrections_using_test_truth": corrections,
                  "result_methods": [controller.results[r]["method"] for r in job["result_ids"]],
                  "training_count": len(result["training_sample_ids"]),
                  "validation_count": len(result["validation_sample_ids"]),
                  "training_position_rmse_mm": result["metrics"]["training"]["position_mm"]["rmse"],
                  "validation_position_rmse_mm": result["metrics"]["validation"]["position_mm"]["rmse"],
                  "hardware_tested": False}
        write_json(output / "desktop-verification.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        window.close()
        controller.pool.shutdown(wait=True)
        rig.close()


if __name__ == "__main__":
    main()

