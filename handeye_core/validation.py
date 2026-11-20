import numpy as np
from .detection import project
from .transforms import inverse, rotation_error_deg


def statistics(values):
    values = np.asarray(values, float)
    if len(values) == 0:
        return None
    return {"mean": float(values.mean()), "rmse": float(np.sqrt(np.mean(values ** 2))),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def evaluate(samples, X, Y_train, camera):
    rows = []
    for sample in samples:
        G, C = np.asarray(sample["G"]), np.asarray(sample["analysis"]["C"])
        Y_i = G @ X @ C
        analysis = sample["analysis"]
        delta_mm = 1000 * (Y_i[:3, 3] - Y_train[:3, 3])
        row = {"sample_id": sample["sample_id"], "role": sample["role"],
               "position_error_mm": float(np.linalg.norm(delta_mm)),
               "position_error_x_mm": float(delta_mm[0]),
               "position_error_y_mm": float(delta_mm[1]),
               "position_error_z_mm": float(delta_mm[2]),
               "rotation_error_deg": rotation_error_deg(Y_train[:3, :3], Y_i[:3, :3]),
               "pnp_rmse_px": analysis.get("pnp_rmse_px"), "predicted_reprojection_rmse_px": None}
        points = np.asarray(analysis["object_points"])
        C_pred = inverse(X) @ inverse(G) @ Y_train
        if np.min((points @ C_pred[:3, :3].T + C_pred[:3, 3])[:, 2]) > 0:
            predicted = project(points, C_pred, camera)
            if np.isfinite(predicted).all():
                row["predicted_reprojection_rmse_px"] = float(np.sqrt(np.mean(np.sum(
                    (predicted - np.asarray(analysis["corners"])) ** 2, axis=1))))
        rows.append(row)
    return {"count": len(rows), "per_sample": rows,
            "position_mm": statistics([r["position_error_mm"] for r in rows]),
            "position_mm_xyz": {axis: statistics([r[f"position_error_{axis}_mm"] for r in rows])
                                for axis in ("x", "y", "z")},
            "rotation_deg": statistics([r["rotation_error_deg"] for r in rows]),
            "predicted_reprojection_px": statistics([r["predicted_reprojection_rmse_px"] for r in rows
                                                     if r["predicted_reprojection_rmse_px"] is not None])}

