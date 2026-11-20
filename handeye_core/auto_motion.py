"""Target poses for the optional automatic repositioning between captures."""
import numpy as np
from scipy.spatial.transform import Rotation
from .errors import HandeyeError
from .transforms import checked, transform


def pivot_distance(radius, max_tilt_deg, max_travel_m):
    """How far in front of the end link the pivot sits; ``radius`` is the distance to the board centre.

    Swinging around the board itself keeps it fixed in the image, but the end link travels
    radius × tilt, which a far board puts out of the arm's reach. The pivot therefore moves
    towards the end link until the largest tilt moves the end by no more than max_travel_m.
    """
    return min(radius, max_travel_m / (2 * np.sin(np.radians(max_tilt_deg) / 2)))


def image_drift_deg(radius, pivot, max_tilt_deg):
    """Largest angle the board centre moves in the image, taking the camera to sit at the end link."""
    tilt = np.radians(max_tilt_deg)
    return np.degrees(tilt - np.arctan2(pivot * np.sin(tilt), radius - pivot * (1 - np.cos(tilt))))


def image_margin_deg(corners, board, camera, max_roll_deg):
    """Angle the detected board can move in the image before the detector loses part of it.

    Covers the squares outside the outermost inner corners, which the detector needs, and the
    roll about the view axis, which turns the board about its centre in the image.
    """
    K = np.array(camera.K)
    cols, rows = board.pattern_size
    grid = np.asarray(corners, float).reshape(rows, cols, 2)
    grid = np.concatenate([2 * grid[:1] - grid[1:2], grid, 2 * grid[-1:] - grid[-2:-1]])
    grid = np.concatenate([2 * grid[:, :1] - grid[:, 1:2], grid, 2 * grid[:, -1:] - grid[:, -2:-1]], axis=1)
    centre = grid.reshape(-1, 2).mean(axis=0)
    margins = []
    for roll in np.radians([-max_roll_deg, 0, max_roll_deg]):
        turn = np.array([[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]])
        rolled = centre + (grid.reshape(-1, 2) - centre) @ turn.T
        for axis, size in ((0, camera.width), (1, camera.height)):
            angle = lambda pixel: np.degrees(np.arctan((pixel - K[axis, 2]) / K[axis, axis]))
            margins += [angle(rolled[:, axis].min()) - angle(0), angle(size) - angle(rolled[:, axis].max())]
    return min(margins)


def sphere_targets(seed, center, count, max_tilt_deg, max_roll_deg, max_travel_m):
    """End poses that swing the seed pose around a pivot on the line to the board centre.

    Each target is the seed rotated rigidly about the pivot (see pivot_distance), so no
    hand-eye estimate is needed. Targets spiral outwards from the seed, so each one is close
    to the one before it.
    """
    seed = checked(seed)
    view = np.asarray(center, float) - seed[:3, 3]
    radius = np.linalg.norm(view)
    if radius < 1e-3:
        raise HandeyeError("AUTO_BOARD_CENTER", "末端和配置的棋盘中心重合，请核对 auto_motion.board_center_m")
    view = view / radius
    pivot = seed[:3, 3] + pivot_distance(radius, max_tilt_deg, max_travel_m) * view
    side = np.cross(view, np.eye(3)[np.argmin(np.abs(view))])
    side /= np.linalg.norm(side)
    up = np.cross(view, side)
    turns = np.sqrt(count / np.pi)  # spacing between turns matches spacing along the spiral
    targets = []
    for i in range(1, count + 1):
        s = np.sqrt(i / count)
        azimuth = 2 * np.pi * turns * s
        tilt = Rotation.from_rotvec(np.radians(max_tilt_deg) * s * (np.cos(azimuth) * side + np.sin(azimuth) * up))
        roll = Rotation.from_rotvec(np.radians(max_roll_deg) * (0, 1, 0, -1)[i % 4] * view)
        R = (tilt * roll).as_matrix()
        targets.append(transform(R @ seed[:3, :3], pivot + R @ (seed[:3, 3] - pivot)))
    return targets
