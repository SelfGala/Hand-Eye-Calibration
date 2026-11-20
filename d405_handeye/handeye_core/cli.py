import argparse
import json
from pathlib import Path
import sys
from .config import load_config
from .errors import HandeyeError


def main(argv=None):
    parser = argparse.ArgumentParser(description="D405 眼在手上标定（原生 Qt / 离线 CLI）")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="检查环境；不会查询机械臂")
    doctor.add_argument("--config")
    doctor.add_argument("--camera", action="store_true", help="实际打开相机测试配置")
    doctor.add_argument("--ros-graph", action="store_true", help="仅发现位姿服务，不调用")
    doctor.add_argument("--output")
    gui = sub.add_parser("gui", help="原生桌面窗口")
    gui.add_argument("--config")
    gui.add_argument("--demo", action="store_true", help="显式合成相机和位姿，不连接硬件")
    gui.add_argument("--session", help="打开保存的会话；离线查看与重算，不连接相机和机械臂")
    gui.add_argument("--resume", nargs="?", const="", default=None,
                     help="继续已有会话并保持相机和位姿可用；省略路径时使用最近一次有样本或缓存的会话")
    gui.add_argument("--smoke-output", help=argparse.SUPPRESS)
    offline = sub.add_parser("offline", help="从保存的数据复算")
    offline.add_argument("--session", required=True)
    offline.add_argument("--method", action="append", choices=["PARK", "TSAI"])
    offline.add_argument("--redetect", action="store_true")
    demo = sub.add_parser("demo", help="生成明确标注的合成测试数据")
    demo.add_argument("--config")
    demo.add_argument("--output", default="./handeye_data")
    demo.add_argument("--train", type=int, default=20)
    demo.add_argument("--validation", type=int, default=5)
    demo.add_argument("--noise-px", type=float, default=0)
    args, unknown = parser.parse_known_args(argv)
    if unknown and args.command not in ("gui", "doctor"):
        parser.error(f"无法识别参数 {unknown}")
    try:
        if args.command == "doctor":
            from .doctor import diagnose
            report = diagnose(load_config(args.config), args.camera, args.ros_graph, unknown)
            text = json.dumps(report, ensure_ascii=False, indent=2)
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text + "\n", encoding="utf-8")
            print(text)
        elif args.command == "gui":
            from handeye_gui.app import launch
            return launch(args, unknown)
        elif args.command == "demo":
            from .synthetic import create_dataset
            from .dataset import write_json
            if min(args.train, args.validation) < 0 or args.train < 3 or args.noise_px < 0:
                raise HandeyeError("DEMO_CONFIG", "合成训练数量至少 3，验证数量和像素噪声非负")
            dataset, truth = create_dataset(args.output, load_config(args.config), args.train, args.validation, args.noise_px)
            write_json(dataset.path / "synthetic_truth.json", {"X": truth.tolist(), "synthetic": True})
            print(str(dataset.path))
        elif args.command == "offline":
            from .dataset import Dataset, read_image
            from .detection import detect
            from .solver import solve
            dataset = Dataset(args.session)
            revision, samples = dataset.snapshot()
            if args.redetect:
                replacements = {}
                for sample in samples:
                    if not sample["enabled"]:
                        continue
                    image = read_image(dataset.path / "samples" / sample["sample_id"] / "image.png")
                    replacements[sample["sample_id"]] = detect(image, dataset.board, dataset.camera,
                                                                sample["analysis"].get("flip_180", False))
                dataset.replace_analyses(replacements, "offline_redetect")
                revision, samples = dataset.snapshot()
            failed = False
            for method in args.method or ["PARK"]:
                try:
                    result = dataset.save_result(solve(samples, dataset.camera, method,
                        dataset.session["config"].get("validation")), revision)
                    print(json.dumps({"method": method, "result_path": str(dataset.path / "results" / result["result_id"]),
                        "validation_state": result["validation_state"], "warnings": result["warnings"]}, ensure_ascii=False))
                except HandeyeError as exc:
                    failed = True
                    print(f"{method}: {exc.code}: {exc}", file=sys.stderr)
            return 1 if failed else 0
    except (HandeyeError, ImportError, OSError, ValueError) as exc:
        print(f"{getattr(exc, 'code', 'ERROR')}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

