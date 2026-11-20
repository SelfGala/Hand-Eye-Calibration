import os
import sys
from pathlib import Path
from handeye_core.config import load_config
from handeye_core.dataset import Dataset
from handeye_core.models import Board
from handeye_core.controller import Controller
from handeye_core.doctor import diagnose
from handeye_core.errors import HandeyeError


def launch(args, ros_args):
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
                                                  or os.environ.get("QT_QPA_PLATFORM") == "offscreen"):
        raise HandeyeError("DISPLAY_UNAVAILABLE", "没有桌面显示环境，请在 Orin 桌面运行 GUI 或使用 offline 命令")
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication
    from .main_window import MainWindow
    app = QApplication.instance() or QApplication([sys.argv[0]])
    from .fonts import configure_fonts
    configure_fonts(app)
    config = load_config(args.config)
    # Motion settings belong to this run; a resumed session only restores what its data depends on.
    auto_motion = config["auto_motion"]
    camera = runtime = controller = source = mover = None
    try:
        resume_path = None
        if args.resume is not None:
            from handeye_core.dataset import resolve_resume
            resume_path = resolve_resume(args.resume, config["storage"]["root"])
        if args.session and resume_path is not None:
            raise HandeyeError("RESUME", "不能同时离线打开和继续标定")
        if args.session:
            dataset = Dataset(args.session)
            config = dataset.session["config"]
            controller = Controller(config, dataset, None, None)
        else:
            if resume_path is not None:
                dataset = Dataset(resume_path)
                saved = dataset.session["config"]
                # The session's config wins below, so a config file that names another arm would be ignored silently.
                if args.config and (saved["frames"] != config["frames"]
                                    or saved["robot"]["pose_source"] != config["robot"]["pose_source"]):
                    raise HandeyeError("RESUME_MISMATCH",
                                       f"会话 {dataset.path.name} 记录的是 {saved['frames']['base']} → {saved['frames']['end']}"
                                       f"（{saved['robot']['pose_source']}），当前配置是 {config['frames']['base']} → "
                                       f"{config['frames']['end']}（{config['robot']['pose_source']}）。"
                                       "换了末端或位姿来源要新建会话：去掉 --resume 再启动")
                config = saved
            if args.demo:
                from handeye_core.synthetic import DemoRig
                camera = DemoRig(Board(**config["board"]))
                source = mover = camera
                auto_motion = {**auto_motion, "board_center_m": camera.board_center().tolist()}
            elif config["robot"]["pose_source"] == "flexiv_sdk":
                from handeye_core.camera import RealSenseCamera
                from handeye_core.flexiv_pose import FlexivPoseSource
                source = FlexivPoseSource(config)
                camera = RealSenseCamera(config["camera"]).start()
            else:
                from handeye_core.camera import RealSenseCamera
                from handeye_core.ros_runtime import RosRuntime
                runtime = RosRuntime(config, ros_args)
                source = runtime.pose_source
                mover = runtime.create_mover(auto_motion)
                camera = RealSenseCamera(config["camera"]).start()
            if resume_path is None:
                dataset = Dataset.create(config["storage"]["root"], config, camera.model, diagnose(config))
            controller = Controller(config, dataset, camera, source, mover, auto_motion)
        if runtime:
            runtime.bind(controller)
        window = MainWindow(controller, args.demo)
        window.show()
        if args.smoke_output:
            def screenshot():
                output = Path(args.smoke_output).resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                if not window.grab().save(str(output)):
                    raise RuntimeError(f"无法保存界面截图 {output}")
                window.close()
                app.quit()
            QTimer.singleShot(2000, screenshot)
        return app.exec_()
    finally:
        if controller:
            controller.close()
            controller.pool.shutdown(wait=True)
        if camera:
            camera.close()
        if runtime is None and source is not None and source is not camera:
            closer = getattr(source, "close", None)
            if closer:
                closer()
        if runtime:
            runtime.close()

