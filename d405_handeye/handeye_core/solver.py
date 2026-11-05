from itertools import combinations
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .errors import HandeyeError
from .transforms import checked, transform, inverse, mean_transform
from .validation import evaluate

METHODS = {"PARK": cv2.CALIB_HAND_EYE_PARK, "TSAI": cv2.CALIB_HAND_EYE_TSAI}


def excitation(poses):
    vectors = [Rotation.from_matrix((inverse(a) @ b)[:3, :3]).as_rotvec()
               for a, b in combinations(poses, 2)]
    if not vectors:
        return {"rank": 0, "singular_values_rad": []}
    singular = np.linalg.svd(np.asarray(vectors), compute_uv=False)
    rank = int(np.sum(singular > max(1e-8, singular[0] * 1e-6)))
    return {"rank": rank, "singular_values_rad": singular.tolist()}


def solve(samples, camera, method="PARK", validation_config=None):
    method = method.upper()
    if method not in METHODS:
        raise HandeyeError("METHOD", f"不支持的算法 {method}")
    train = [s for s in samples if s.get("enabled", True) and s["role"] == "train"]
    validation = [s for s in samples if s.get("enabled", True) and s["role"] == "validation"]
    if len(train) < 3:
        raise HandeyeError("SAMPLES_INSUFFICIENT", "至少需要 3 个不同的训练姿态")
    G = [checked(s["G"]) for s in train]
    C = [checked(s["analysis"]["C"]) for s in train]
    coverage = excitation(G)
    if coverage["rank"] < 2 or excitation(C)["rank"] < 2:
        raise HandeyeError("DEGENERATE_MOTION", "旋转运动退化：请增加至少两个非平行轴的旋转姿态")
    try:
        R, t = cv2.calibrateHandEye([g[:3, :3] for g in G], [g[:3, 3:4] for g in G],
                                    [c[:3, :3] for c in C], [c[:3, 3:4] for c in C],
                                    method=METHODS[method])
        X = transform(R, t)
    except (cv2.error, HandeyeError) as exc:
        raise HandeyeError("SOLVE_FAILED", f"{method} 求解失败：{exc}") from exc
    Y = mean_transform([g @ X @ c for g, c in zip(G, C)])
    training_metrics = evaluate(train, X, Y, camera)
    validation_metrics = evaluate(validation, X, Y, camera)
    warnings = []
    if len(train) < 15:
        warnings.append("训练样本少于建议的 15 组，可继续增加不同朝向；当前结果仍可导出")
    for name, metrics in (("训练", training_metrics), ("验证", validation_metrics)):
        config = validation_config or {}
        if metrics["count"]:
            if metrics["position_mm"]["rmse"] > config.get("warning_position_rmse_mm", 2):
                warnings.append(f"{name}位置误差较大，请核对角点方向、位姿方向、板尺寸与配对")
            if metrics["rotation_deg"]["rmse"] > config.get("warning_rotation_rmse_deg", 0.5):
                warnings.append(f"{name}旋转误差较大，请核对物理原点与坐标关系")
    return {"schema_version": 2, "calibration_type": "eye_in_hand", "method": method,
            "transform_convention": "p_parent = T_parent_child @ p_child", "translation_unit": "m",
            "quaternion_order": "xyzw", "matrix_order": "row_major", "T_parent_child": X.tolist(),
            "translation": X[:3, 3].tolist(), "quaternion": Rotation.from_matrix(X[:3, :3]).as_quat().tolist(),
            "T_base_board_training_mean": Y.tolist(), "training_sample_ids": [s["sample_id"] for s in train],
            "validation_sample_ids": [s["sample_id"] for s in validation],
            "has_validation_metrics": bool(validation),
            "validation_state": "validation_available" if validation else "computed",
            "pairing_mode": "manual_stationary_snapshot", "robot_read_policy": "capture_only",
            "pose_reads_per_capture": 1, "excitation": coverage, "warnings": warnings,
            "metrics": {"training": training_metrics, "validation": validation_metrics,
                        "statistics": "per-pose Euclidean translation / SO(3) angle; p95 linear percentile",
                        "pixel_error": "prediction from X and training-only board mean; no validation refit"}}

