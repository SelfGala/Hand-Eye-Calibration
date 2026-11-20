from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import json
import shutil
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QColor
from PyQt5.QtWidgets import (QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QSplitter, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QTextEdit, QTabWidget, QFileDialog, QMessageBox, QHeaderView)
from handeye_core.models import Board
from handeye_core.detection import detect, overlay
from handeye_core.dataset import read_image, previous_session
from handeye_core.errors import HandeyeError
from .workflow import guidance


def pixmap(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.ascontiguousarray(image)
    qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


class MainWindow(QMainWindow):
    def __init__(self, controller, demo=False):
        super().__init__()
        self.controller, self.demo = controller, demo
        self.preview_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview-detection")
        self.preview_future = None
        self.current_job = None
        self.last_revision = -1
        self.last_session_path = str(controller.dataset.path)
        self.viewing_sample = False
        self.displayed_image = None
        self.selected_result_id = ""
        self.setWindowTitle("D405 眼在手上标定" + (" — 合成演示（不连接硬件）" if demo else ""))
        self.resize(1380, 940)
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        board_line = QHBoxLayout()
        self.interpretation = QComboBox()
        self.interpretation.addItem("计数指内角点", "inner_corners")
        self.interpretation.addItem("计数指方格数", "squares")
        self.cols, self.rows = QSpinBox(), QSpinBox()
        for spin in (self.cols, self.rows):
            spin.setRange(3, 30)
        self.square = QDoubleSpinBox()
        self.square.setRange(0.1, 1000)
        self.square.setDecimals(3)
        self.square.setSuffix(" mm")
        config = controller.dataset.board
        self.interpretation.setCurrentIndex(0 if config.count_interpretation == "inner_corners" else 1)
        self.cols.setValue(config.cols)
        self.rows.setValue(config.rows)
        self.square.setValue(config.square_length_m * 1000)
        self.new_button = QPushButton("按参数新建会话")
        self.new_button.clicked.connect(self.new_session)
        self.resume_button = QPushButton("继续上次标定")
        self.resume_button.clicked.connect(self.resume_previous)
        for w in (QLabel("棋盘"), self.interpretation, self.cols, QLabel("×"), self.rows,
                  QLabel("单格边长"), self.square, self.new_button, self.resume_button):
            board_line.addWidget(w)
        board_line.addStretch()
        layout.addLayout(board_line)
        self.board_info = QLabel()
        layout.addWidget(self.board_info)
        for signal in (self.interpretation.currentIndexChanged, self.cols.valueChanged,
                       self.rows.valueChanged, self.square.valueChanged):
            signal.connect(self.describe_board)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        capture_page = QWidget()
        capture_layout = QVBoxLayout(capture_page)
        splitter = QSplitter(Qt.Horizontal)
        capture_layout.addWidget(splitter, 3)
        self.image = QLabel("相机图像将在这里显示")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(480, 280)
        self.image.setStyleSheet("background:#20242b;color:#ddd")
        splitter.addWidget(self.image)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        self.guide = QLabel()
        self.guide.setWordWrap(True)
        side_layout.addWidget(self.guide)
        self.diagram = QLabel()
        side_layout.addWidget(self.diagram)
        self.detection_info = QLabel("等待图像")
        self.detection_info.setWordWrap(True)
        side_layout.addWidget(self.detection_info)
        self.show_overlay = QCheckBox("显示角点、原点与坐标轴")
        self.show_overlay.setChecked(True)
        side_layout.addWidget(self.show_overlay)
        self.flip = QCheckBox("本次角点编号旋转 180°")
        side_layout.addWidget(self.flip)
        self.live_button = QPushButton("返回实时预览")
        self.live_button.clicked.connect(self.resume_preview)
        side_layout.addWidget(self.live_button)
        side_layout.addStretch()
        splitter.addWidget(side)
        splitter.setSizes([900, 360])
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["缩略图", "样本", "角色", "启用", "角点", "PnP / px", "查询 / ms", "配对", "编号"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setIconSize(QSize(90, 50))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.cellClicked.connect(self.view_sample)
        capture_layout.addWidget(self.table, 2)
        self.tabs.addTab(capture_page, "采集与样本")
        result_page = QWidget()
        result_layout = QVBoxLayout(result_page)
        self.result_combo = QComboBox()
        self.result_combo.currentIndexChanged.connect(self.display_result)
        result_layout.addWidget(self.result_combo)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlainText("尚未计算。没有独立验证数据时显示“未验证”。")
        result_layout.addWidget(self.result_text)
        self.tabs.addTab(result_page, "标定结果")
        actions = QHBoxLayout()
        self.role = QComboBox()
        self.role.addItem("训练样本", "train")
        self.role.addItem("验证样本", "validation")
        self.label = QLineEdit()
        self.label.setPlaceholderText("可选样本备注")
        self.capture_button = QPushButton("拍照 [空格]")
        self.capture_button.clicked.connect(self.capture)
        self.cancel_button = QPushButton("取消当前任务")
        self.cancel_button.clicked.connect(controller.cancel)
        self.enabled_button = QPushButton("启用 / 停用选中")
        self.enabled_button.clicked.connect(self.toggle_sample)
        self.role_button = QPushButton("切换选中角色")
        self.role_button.clicked.connect(self.toggle_role)
        self.flip_button = QPushButton("纠正选中编号 180°")
        self.flip_button.clicked.connect(self.flip_sample)
        self.compute_button = QPushButton("计算 / 验证")
        self.compute_button.clicked.connect(self.compute)
        self.export_button = QPushButton("导出结果")
        self.export_button.clicked.connect(self.export)
        for w in (self.role, self.label, self.capture_button, self.cancel_button, self.enabled_button,
                  self.role_button, self.flip_button, self.compute_button, self.export_button):
            actions.addWidget(w)
        layout.addLayout(actions)
        self.task_status = QLabel("准备就绪")
        self.task_status.setWordWrap(True)
        layout.addWidget(self.task_status)
        self.describe_board()
        self.refresh_samples()
        self._announce_previous()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(max(30, int(1000 / controller.config["ui"]["preview_fps"])))

    def board_config(self):
        return {"count_interpretation": self.interpretation.currentData(), "cols": self.cols.value(),
                "rows": self.rows.value(), "square_length_m": self.square.value() / 1000,
                "physical_origin": self.controller.dataset.board.physical_origin}

    def describe_board(self):
        try:
            board = Board(**self.board_config())
            cols, rows = board.pattern_size
            width = (cols + 1) * board.square_length_m * 1000
            height = (rows + 1) * board.square_length_m * 1000
            camera = self.controller.dataset.camera
            z_hint = max(camera.K[0][0] * width / 1000 / (0.8 * camera.width),
                         camera.K[1][1] * height / 1000 / (0.8 * camera.height))
            changed = board.to_dict() != self.controller.dataset.board.to_dict()
            self.board_info.setText(f"内角点 {cols}×{rows} = {cols * rows} 个；方格区约 {width:.1f}×{height:.1f} mm"
                                    + f"；正对取景起始距离约 {z_hint:.2f} m，以实际画面为准"
                                    + ("；参数已修改，点击“按参数新建会话”后生效" if changed else ""))
            scale = min(200 / (cols + 1), 125 / (rows + 1))
            canvas = QPixmap(230, 155)
            canvas.fill(QColor("#e8e8e8"))
            painter = QPainter(canvas)
            for y in range(rows + 1):
                for x in range(cols + 1):
                    painter.fillRect(int(12 + x * scale), int(12 + y * scale), int(scale + 1), int(scale + 1),
                                     QColor("#171717" if (x + y) % 2 == 0 else "white"))
            painter.setPen(QColor("#00a040"))
            painter.drawEllipse(int(12 + scale - 4), int(12 + scale - 4), 8, 8)
            painter.drawText(12, 150, "0: 同一物理角  /  X →  Y ↓")
            painter.end()
            self.diagram.setPixmap(canvas)
        except Exception as exc:
            self.board_info.setText(str(exc))

    def run_action(self, function):
        try:
            function()
        except Exception as exc:
            self.task_status.setText(f"{getattr(exc, 'code', 'ERROR')}: {exc}")

    def new_session(self):
        def action():
            board = self.board_config()
            Board(**board)
            self.controller.new_session(board)
            if self.demo:
                self.controller.camera.board = self.controller.dataset.board
                self.controller.camera._image_index = -1
            self.result_combo.clear()
            self.selected_result_id = ""
            self.result_text.setPlainText("新会话，尚未计算")
            self.viewing_sample = False
            self.refresh_samples()
            self.describe_board()
            self.task_status.setText("新会话已创建；原会话保留在磁盘，可点“继续上次标定”回去")
        self.run_action(action)

    def sync_board_widgets(self):
        board = self.controller.dataset.board
        widgets = (self.interpretation, self.cols, self.rows, self.square)
        for widget in widgets:
            widget.blockSignals(True)
        self.interpretation.setCurrentIndex(0 if board.count_interpretation == "inner_corners" else 1)
        self.cols.setValue(board.cols)
        self.rows.setValue(board.rows)
        self.square.setValue(board.square_length_m * 1000)
        for widget in widgets:
            widget.blockSignals(False)
        self.describe_board()

    def resume_previous(self):
        def action():
            path = previous_session(self.controller.config["storage"]["root"], self.controller.dataset.path)
            report = self.controller.resume_session(path)
            if self.demo and self.controller.camera is not None:
                self.controller.camera.board = self.controller.dataset.board
                self.controller.camera._image_index = -1
            self.sync_board_widgets()
            self.viewing_sample = False
            self.last_session_path = str(self.controller.dataset.path)
            with self.controller.lock:
                results = list(self.controller.results.items())
            self.result_combo.blockSignals(True)
            self.result_combo.clear()
            for result_id, result in results:
                self.result_combo.addItem(result["method"] + " · " + result_id, result_id)
            if results:
                self.result_combo.setCurrentIndex(len(results) - 1)
            self.result_combo.blockSignals(False)
            self.selected_result_id = ""
            self.last_revision = -1
            self.refresh_samples()
            if not results:
                self.result_text.setPlainText("已继续会话，尚未计算")
            recovered, failed = len(report["recovered"]), len(report["failed"])
            text = f"已继续 {path.name}，当前 {len(self.samples)} 组样本"
            if recovered:
                text += f"，从缓存恢复 {recovered} 组"
            if failed:
                text += f"，仍有 {failed} 组缓存在 cache/pending（识别未通过）"
            self.task_status.setText(text)
        self.run_action(action)

    def _announce_previous(self):
        try:
            path = previous_session(self.controller.config["storage"]["root"], self.controller.dataset.path)
        except HandeyeError:
            return
        self.task_status.setText(f"准备就绪；可继续上次标定 {path.name}")

    def capture(self):
        def action():
            if self.board_config() != self.controller.dataset.board.to_dict():
                raise ValueError("棋盘参数已修改，请先按参数新建会话")
            self.current_job = self.controller.capture(self.role.currentData(), self.label.text(), self.flip.isChecked())
            self.viewing_sample = False
        self.run_action(action)

    def compute(self):
        self.run_action(lambda: setattr(self, "current_job", self.controller.compute()))

    def selected_sample(self):
        row = self.table.currentRow()
        if row < 0:
            raise ValueError("请先选中一个样本")
        return self.samples[row]

    def toggle_sample(self):
        def action():
            sample = self.selected_sample()
            self.controller.set_sample(sample["sample_id"], enabled=not sample["enabled"])
            self.refresh_samples()
        self.run_action(action)

    def toggle_role(self):
        def action():
            sample = self.selected_sample()
            self.controller.set_sample(sample["sample_id"], role="validation" if sample["role"] == "train" else "train")
            self.refresh_samples()
        self.run_action(action)

    def flip_sample(self):
        def action():
            sample_id = self.selected_sample()["sample_id"]
            self.controller.flip_sample(sample_id)
            self.refresh_samples()
            for index, sample in enumerate(self.samples):
                if sample["sample_id"] == sample_id:
                    self.table.selectRow(index)
                    self.view_sample(index, 0)
        self.run_action(action)

    def refresh_samples(self):
        self.last_revision, self.samples = self.controller.dataset.snapshot()
        self.table.setRowCount(len(self.samples))
        for row, sample in enumerate(self.samples):
            analysis = sample["analysis"]
            query = sample.get("robot", {}).get("roundtrip_ms")
            values = ["", sample["sample_id"], "训练" if sample["role"] == "train" else "验证",
                      "是" if sample["enabled"] else "否", str(analysis["corner_count"]),
                      f"{analysis['pnp_rmse_px']:.3f}", f"{query:.1f}" if query is not None else "未知",
                      "人工单快照", "180°" if analysis.get("flip_180") else "原始"]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))
            image_path = self.controller.dataset.path / "samples" / sample["sample_id"] / "image.png"
            icon = QIcon(str(image_path))
            self.table.item(row, 0).setIcon(icon)
            self.table.setRowHeight(row, 56)
        self.guide.setText(guidance(self.samples, self.controller.config["calibration"]["recommended_training_samples"]))
        pending = self.controller.dataset.pending_captures()
        camera = self.controller.dataset.camera
        origin = "合成演示" if self.demo else ("离线查看" if self.controller.camera is None else "RealSense SDK")
        source = self.samples[-1].get("robot", {}).get("source_kind", "尚未采样") if self.samples else "尚未采样"
        source = {"replay": "模拟 / 回放", "sdk_on_request": "本次 SDK 读取", "topic_compat": "状态话题",
                  "tf_compat": "本轮 TF", "existing_driver_snapshot": "驱动快照"}.get(source, source)
        pending_text = f" | 待恢复缓存 {len(pending)}" if pending else ""
        self.info.setText(f"{origin} | {camera.width}×{camera.height} | 按需读取 | 数据版本 {self.last_revision} | 最近位姿来源 {source}{pending_text}\n"
                          f"会话：{self.controller.dataset.path}")
        self.display_result()

    def view_sample(self, row, col):
        def action():
            sample = self.samples[row]
            image = read_image(self.controller.dataset.path / "samples" / sample["sample_id"] / "image.png")
            if self.show_overlay.isChecked():
                image = overlay(image, sample["analysis"], self.controller.dataset.board, self.controller.dataset.camera)
            self.viewing_sample = True
            self.set_image(image)
            self.detection_info.setText(f"查看已保存样本\n{sample['sample_id']}\n原始图像和测量记录保持不变")
        self.run_action(action)

    def resume_preview(self):
        self.viewing_sample = False

    def set_image(self, image):
        self.displayed_image = image
        self.image.setPixmap(pixmap(image).scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def tick(self):
        controller = self.controller
        if str(controller.dataset.path) != self.last_session_path:
            self.last_session_path = str(controller.dataset.path)
            self.result_combo.clear()
            self.selected_result_id = ""
            self.last_revision = -1
            self.viewing_sample = False
        busy = bool(controller.active_job)
        self.capture_button.setEnabled(not busy and controller.camera is not None)
        for button in (self.new_button, self.resume_button, self.enabled_button, self.role_button,
                       self.flip_button, self.compute_button):
            button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        if self.current_job:
            job = controller.get_job(self.current_job)
            state = {"queued": "已受理", "running": "处理中", "succeeded": "完成", "failed": "失败", "cancelled": "已取消"}[job["state"]]
            self.task_status.setText(f"{state} | 本轮位姿请求 {job['pose_request_count']} 次 | "
                                     f"{job['error_code']} {job['message']}")
            if job["state"] in ("succeeded", "failed", "cancelled"):
                if job["result_ids"]:
                    for result_id in job["result_ids"]:
                        self.result_combo.addItem(controller.results[result_id]["method"] + " · " + result_id, result_id)
                    self.result_combo.setCurrentIndex(self.result_combo.count() - len(job["result_ids"]))
                    self.tabs.setCurrentIndex(1)
                    self.display_result()
                self.current_job = None
        if controller.dataset.revision != self.last_revision:
            self.refresh_samples()
        # Include jobs submitted through ROS, using exactly the same controller.
        known = {self.result_combo.itemData(i) for i in range(self.result_combo.count())}
        with controller.lock:
            result_items = list(controller.results.items())
        for result_id, result in result_items:
            if result_id not in known:
                self.result_combo.addItem(result["method"] + " · " + result_id, result_id)
        if self.viewing_sample or controller.camera is None:
            return
        if self.preview_future is not None and self.preview_future.done():
            try:
                image, text, detected = self.preview_future.result()
                self.detection_info.setText(text)
                # 相机一直在出新帧，不能要求帧号相同，否则角点和坐标轴永远画不上去。
                self._overlay = image if detected else None
            except Exception as exc:
                self.detection_info.setText(str(exc))
                self._overlay = None
            self.preview_future = None
        frame = controller.camera.latest()
        if frame is None:
            error = getattr(controller.camera, "error", None)
            self.detection_info.setText(error or "等待首帧")
            if error:
                self.image.setText(error)
            return
        if self.show_overlay.isChecked() and getattr(self, "_overlay", None) is not None:
            self.set_image(self._overlay)
        else:
            self.set_image(frame.image)
        if self.preview_future is None:
            board, camera = controller.dataset.board, controller.dataset.camera
            flip, show = self.flip.isChecked(), self.show_overlay.isChecked()
            def process():
                try:
                    analysis = detect(frame.image, board, camera, flip)
                    image = overlay(frame.image, analysis, board, camera) if show else frame.image
                    text = (f"角点 {analysis['corner_count']} · PnP {analysis['pnp_rmse_px']:.3f} px\n"
                            "核对 0 点；保持不动后点击拍照")
                    return image, text, True
                except Exception as exc:
                    return frame.image, str(exc), False
            self.preview_future = self.preview_pool.submit(process)

    def display_result(self, *args):
        result_id = self.result_combo.currentData()
        if not result_id:
            return
        result = self.controller.get_result(result_id)
        self.selected_result_id = result_id
        text = f"{result['method']} | 数据版本 {result['dataset_revision']}"
        if result["stale"]:
            text += " · 已有样本变更，请重新计算"
        text += f"\n{result['parent_frame']} <- {result['child_frame']}\n平移 / m: {result['translation']}\n"
        text += f"四元数 / xyzw: {result['quaternion']}\n"
        text += "矩阵:\n" + np.array2string(np.asarray(result["T_parent_child"]), precision=7) + "\n\n"
        for key, title in (("training", "训练"), ("validation", "独立验证")):
            metrics = result["metrics"][key]
            if not metrics["count"]:
                text += f"{title}：未验证\n"
                continue
            axes = metrics.get("position_mm_xyz") or {}
            axis_text = ""
            if all(axes.get(name) for name in ("x", "y", "z")):
                axis_text = "（" + "，".join(
                    f"{name.upper()} {axes[name]['rmse']:.4f}" for name in ("x", "y", "z")) + "）"
            text += (f"{title} {metrics['count']} 组：位置 RMSE {metrics['position_mm']['rmse']:.4f} mm{axis_text}，"
                     f"旋转 RMSE {metrics['rotation_deg']['rmse']:.4f}°\n")
            if metrics["predicted_reprojection_px"]:
                text += f"  手眼链预测像素 RMSE {metrics['predicted_reprojection_px']['rmse']:.4f} px\n"
        text += "\n" + "\n".join(result["warnings"]) + "\n\n完整报告:\n" + json.dumps(result, ensure_ascii=False, indent=2)
        self.result_text.setPlainText(text)

    def export(self):
        def action():
            result = self.controller.get_result(self.selected_result_id)
            target = QFileDialog.getExistingDirectory(self, "选择导出目录")
            if not target:
                return
            destination = Path(target) / result["result_id"]
            # Export to a unique result directory; no existing files are overwritten.
            shutil.copytree(self.controller.dataset.path / "results" / result["result_id"], destination)
            self.task_status.setText(f"已导出：{destination}" + ("；这是旧数据版本的结果" if result["stale"] else ""))
        self.run_action(action)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if not isinstance(self.focusWidget(), QLineEdit) and self.capture_button.isEnabled():
                self.capture()
                event.accept()
                return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        self.controller.close()
        self.preview_pool.shutdown(wait=False, cancel_futures=True)
        event.accept()

