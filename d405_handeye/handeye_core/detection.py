import cv2
import numpy as np
from .errors import HandeyeError
from .transforms import transform, checked
from .camera_geometry import normalized_pixels, project_camera_points


def project(points, C, camera):
    C = checked(C)
    camera_points = np.asarray(points, np.float64) @ C[:3, :3].T + C[:3, 3]
    return project_camera_points(camera_points, camera)


def from_corners(raw, board, camera, flip=False):
    raw = np.asarray(raw, np.float64).reshape(-1, 2)
    objects = board.object_points()
    if raw.shape != (len(objects), 2) or not np.isfinite(raw).all():
        raise HandeyeError("CHESSBOARD_INCOMPLETE", "未检测到完整棋盘，请调整取景")
    permutation = np.arange(len(raw))
    if flip:
        permutation = permutation[::-1]
    corners = np.ascontiguousarray(raw[permutation])
    K, D = np.array(camera.K), np.array(camera.D)
    fit_corners = corners
    coordinate_space = "original_pixels_opencv_model"
    if camera.distortion_model == "inverse_brown_conrady":
        fit_corners = np.ascontiguousarray(normalized_pixels(corners, camera))
        K, D = np.eye(3), np.zeros(5)
        coordinate_space = "sdk_normalized_rays_no_further_distortion"
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(objects, fit_corners, K, D, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    if ok:
        for rvec, tvec in zip(rvecs, tvecs):
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue
            C = transform(cv2.Rodrigues(rvec)[0], tvec)
            if np.min((objects @ C[:3, :3].T + C[:3, 3])[:, 2]) <= 0:
                continue
            error = np.sqrt(np.mean(np.sum((project(objects, C, camera) - corners) ** 2, axis=1)))
            candidates.append((float(error), rvec, tvec))
    if not candidates:
        raise HandeyeError("PNP_FAILED", "PnP 无有效正深度解")
    candidates.sort(key=lambda p: p[0])
    error, rvec, tvec = candidates[0]
    rvec, tvec = cv2.solvePnPRefineLM(objects, fit_corners, K, D, rvec.copy(), tvec.copy())
    C = transform(cv2.Rodrigues(rvec)[0], tvec)
    if np.min((objects @ C[:3, :3].T + C[:3, 3])[:, 2]) <= 0:
        raise HandeyeError("PNP_FAILED", "精修后的棋盘在相机后方")
    error = float(np.sqrt(np.mean(np.sum((project(objects, C, camera) - corners) ** 2, axis=1))))
    area = float(cv2.contourArea(cv2.convexHull(corners.astype(np.float32))))
    return {"C": C.tolist(), "raw_corners": raw.tolist(), "corners": corners.tolist(),
            "permutation": permutation.tolist(), "flip_180": bool(flip),
            "object_points": objects.tolist(), "physical_origin": board.physical_origin,
            "corner_count": len(raw), "pnp_rmse_px": error,
            "hull_area_ratio": area / (camera.width * camera.height),
            "positive_depth": True,
            "pnp_coordinate_space": coordinate_space,
            "ippe_candidate_rmse_px": [c[0] for c in candidates],
            "orientation_verified": "manual_visual_check_required"}


def detect(image, board, camera, flip=False):
    if image.shape[:2] != (camera.height, camera.width):
        raise HandeyeError("IMAGE_GEOMETRY", "原图尺寸与活动相机内参不一致")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, board.pattern_size,
                                               flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, board.pattern_size)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    if not found:
        raise HandeyeError("CHESSBOARD_INCOMPLETE", "未完整识别棋盘；请检查 9×6 定义和取景")
    result = from_corners(corners, board, camera, flip)
    result["sharpness_laplacian_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return result


def overlay(image, analysis, board, camera):
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    corners = np.asarray(analysis["corners"], np.float32)
    cv2.drawChessboardCorners(out, board.pattern_size, corners.reshape(-1, 1, 2), True)
    for index in (0, len(corners) - 1):
        p = tuple(np.round(corners[index]).astype(int))
        cv2.circle(out, p, 7, (0, 255, 0), 2)
        cv2.putText(out, str(index), p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 50, 255), 2)
    axes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float) * board.square_length_m * 2
    pixels = project(axes, np.asarray(analysis["C"]), camera)
    if np.isfinite(pixels).all() and np.max(np.abs(pixels)) < 1e7:
        points = np.round(pixels).astype(int)
        for index, name, color in ((1, "X", (0, 0, 255)), (2, "Y", (0, 200, 0)), (3, "Z", (255, 0, 0))):
            cv2.line(out, tuple(points[0]), tuple(points[index]), color, 2)
            cv2.putText(out, name, tuple(points[index]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out

