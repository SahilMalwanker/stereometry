"""Paired checkerboard calibration with per-view diagnostics."""

from pathlib import Path

import cv2
import numpy as np

from .rig import StereoRig


def checkerboard_points(columns, rows, square_mm):
    if columns < 2 or rows < 2 or square_mm <= 0 or not np.isfinite(square_mm):
        raise ValueError("board needs at least 2x2 inner corners and a positive square size")
    points = np.zeros((rows * columns, 3), np.float32)
    points[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square_mm
    return points


def calibrate_points(world_views, left_views, right_views, image_size):
    if len(world_views) < 6 or not len(world_views) == len(left_views) == len(right_views):
        raise ValueError("at least six corresponding board views are required")
    world = [np.asarray(view, np.float32).reshape(-1, 3) for view in world_views]
    left = [np.asarray(view, np.float32).reshape(-1, 2) for view in left_views]
    right = [np.asarray(view, np.float32).reshape(-1, 2) for view in right_views]
    for objects, left_pixels, right_pixels in zip(world, left, right, strict=True):
        if len(objects) < 6 or not len(objects) == len(left_pixels) == len(right_pixels):
            raise ValueError("each view must have matching object and image points")
        if not all(np.isfinite(view).all() for view in (objects, left_pixels, right_pixels)):
            raise ValueError("calibration observations must be finite")
    left_rms, left_matrix, left_dist, left_rotations, left_translations = cv2.calibrateCamera(
        world, left, image_size, None, None
    )
    right_rms, right_matrix, right_dist, right_rotations, right_translations = cv2.calibrateCamera(
        world, right, image_size, None, None
    )
    result = cv2.stereoCalibrate(
        world,
        left,
        right,
        left_matrix,
        left_dist,
        right_matrix,
        right_dist,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7),
    )
    rms, left_matrix, left_dist, right_matrix, right_dist, rotation, translation, _, _ = result
    rig = StereoRig(
        left_matrix, left_dist, right_matrix, right_dist, rotation, translation.ravel(), image_size
    )

    def view_errors(pixels, matrix, distortion, rotations, translations):
        errors = []
        for index, points in enumerate(world):
            projected = cv2.projectPoints(
                points, rotations[index], translations[index], matrix, distortion
            )[0].reshape(-1, 2)
            errors.append(float(np.sqrt(np.mean(np.sum((projected - pixels[index]) ** 2, axis=1)))))
        return errors

    diagnostics = {
        "stereo_rms_px": float(rms),
        "left_rms_px": float(left_rms),
        "right_rms_px": float(right_rms),
        "views": len(world),
        "baseline_mm": rig.baseline_mm,
        "left_view_rms_px": view_errors(
            left, left_matrix, left_dist, left_rotations, left_translations
        ),
        "right_view_rms_px": view_errors(
            right, right_matrix, right_dist, right_rotations, right_translations
        ),
    }
    return rig, diagnostics


def load_gray(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode image {path}")
    return image


def calibrate_folders(left_folder, right_folder, columns, rows, square_mm):
    """PNG pairs must share filenames; rejected detections stay in the diagnostics."""
    left_files = {path.name: path for path in Path(left_folder).glob("*.png")}
    right_files = {path.name: path for path in Path(right_folder).glob("*.png")}
    if not left_files or left_files.keys() != right_files.keys():
        raise ValueError("left/right folders must contain the same nonempty set of PNG filenames")
    world = checkerboard_points(columns, rows, square_mm)
    objects, left_views, right_views, rejected, accepted = [], [], [], [], []
    image_size = None
    for name in sorted(left_files):
        left, right = load_gray(left_files[name]), load_gray(right_files[name])
        size = left.shape[::-1]
        if left.shape != right.shape or (image_size is not None and size != image_size):
            raise ValueError("every frame must have identical dimensions")
        image_size = size
        found_left, points_left = cv2.findChessboardCornersSB(left, (columns, rows))
        found_right, points_right = cv2.findChessboardCornersSB(right, (columns, rows))
        if not found_left or not found_right:
            rejected.append(name)
            continue
        objects.append(world.copy())
        left_views.append(points_left)
        right_views.append(points_right)
        accepted.append(name)
    rig, diagnostics = calibrate_points(objects, left_views, right_views, image_size)
    diagnostics.update(
        accepted_pairs=accepted,
        rejected_pairs=rejected,
        square_mm=square_mm,
        inner_corners=[columns, rows],
    )
    return rig, diagnostics
