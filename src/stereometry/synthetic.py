"""Seeded, explicitly synthetic fixtures; no camera calibration is bundled."""

from pathlib import Path

import cv2
import numpy as np

from .calibration import checkerboard_points
from .rig import StereoRig


def calibration_observations(seed=7, views=18, noise_px=0.08):
    rng = np.random.default_rng(seed)
    matrix = np.array([[700.0, 0, 320], [0, 705, 240], [0, 0, 1]])
    distortion = np.array([-0.07, 0.015, 0.0003, -0.0002, 0.0])
    rotation = cv2.Rodrigues(np.array([0.004, -0.008, 0.002]))[0]
    truth = StereoRig(
        matrix, distortion, matrix.copy(), distortion.copy(), rotation, [-63, 0.4, 0.3], (640, 480)
    )
    world = checkerboard_points(9, 6, 18)
    objects, left, right = [], [], []
    for _ in range(views):
        board_rotation = rng.uniform(-0.45, 0.45, 3)
        translation = np.array(
            [rng.uniform(-160, 30), rng.uniform(-110, 30), rng.uniform(470, 900)]
        )
        board_matrix = cv2.Rodrigues(board_rotation)[0]
        right_rotation = cv2.Rodrigues(rotation @ board_matrix)[0]
        right_translation = rotation @ translation + truth.translation_mm
        left_points = cv2.projectPoints(world, board_rotation, translation, matrix, distortion)[0]
        right_points = cv2.projectPoints(
            world, right_rotation, right_translation, matrix, distortion
        )[0]
        objects.append(world.copy())
        left.append(left_points + rng.normal(0, noise_px, left_points.shape))
        right.append(right_points + rng.normal(0, noise_px, right_points.shape))
    return truth, objects, left, right


def layered_scene(seed=12):
    """Three rectified textured planes, with z-buffered right-view occlusions."""
    rng = np.random.default_rng(seed)
    height, width = 288, 480
    matrix = np.array([[400.0, 0, width / 2], [0, 400, height / 2], [0, 0, 1]])
    rig = StereoRig(
        matrix, np.zeros(5), matrix.copy(), np.zeros(5), np.eye(3), [-63, 0, 0], (width, height)
    )
    texture = rng.integers(20, 236, (height, width), dtype=np.uint8)
    left = cv2.GaussianBlur(texture, (3, 3), 0.65)
    right = np.zeros_like(left)
    disparities = np.empty((height, width), np.float32)
    disparities[:, :160] = 18
    disparities[:, 160:320] = 28
    disparities[:, 320:] = 42
    for disparity in (18, 28, 42):
        row, column = np.nonzero(disparities == disparity)
        visible = column - disparity >= 0
        right[row[visible], column[visible] - disparity] = left[row[visible], column[visible]]
    return rig, left, right, disparities


def _board_texture(columns, rows, square_mm, margin_mm, px_per_mm):
    origin = -square_mm - margin_mm
    width = round(((columns + 1) * square_mm + 2 * margin_mm) * px_per_mm)
    height = round(((rows + 1) * square_mm + 2 * margin_mm) * px_per_mm)
    x_mm = origin + (np.arange(width) + 0.5) / px_per_mm
    y_mm = origin + (np.arange(height) + 0.5) / px_per_mm
    column = np.floor(x_mm / square_mm).astype(int) + 1
    row = np.floor(y_mm / square_mm).astype(int) + 1
    inside_x = (column >= 0) & (column <= columns)
    inside_y = (row >= 0) & (row <= rows)
    dark = ((row[:, None] + column[None, :]) % 2 == 0) & inside_y[:, None] & inside_x[None, :]
    texture = np.where(dark, 25, 230).astype(np.uint8)
    shift = origin + 0.5 / px_per_mm
    to_mm = np.array([[1 / px_per_mm, 0, shift], [0, 1 / px_per_mm, shift], [0, 0, 1]])
    return texture, to_mm


def render_checkerboard_pairs(folder, pairs=10, seed=3, columns=9, rows=6, square_mm=18.0):
    """Write anti-aliased left/ and right/ PNG pairs of a posed board; return the true rig."""
    rng = np.random.default_rng(seed)
    width, height = 640, 480
    matrix = np.array([[620.0, 0, 320], [0, 620, 240], [0, 0, 1]])
    rotation = cv2.Rodrigues(np.array([0.0, 0.01, 0.0]))[0]
    truth = StereoRig(
        matrix, np.zeros(5), matrix.copy(), np.zeros(5), rotation, [-63, 0, 0], (width, height)
    )
    texture, to_mm = _board_texture(columns, rows, square_mm, 2 * square_mm, 4.0)
    center = np.array([(columns - 1) * square_mm / 2, (rows - 1) * square_mm / 2, 0])
    # 2x render whose INTER_AREA downsample lands exactly on the full-resolution pixel grid
    supersample = np.array([[2.0, 0, 0.5], [0, 2.0, 0.5], [0, 0, 1]])
    folders = [Path(folder) / "left", Path(folder) / "right"]
    for path in folders:
        path.mkdir(parents=True, exist_ok=True)
    for index in range(pairs):
        board = cv2.Rodrigues(rng.uniform([-0.4, -0.4, -0.25], [0.4, 0.4, 0.25]))[0]
        position = rng.uniform([-60, -45, 520], [60, 45, 760])
        translation = position - board @ center
        poses = [
            (board, translation),
            (rotation @ board, rotation @ translation + truth.translation_mm),
        ]
        for path, (pose_rotation, pose_translation) in zip(folders, poses, strict=True):
            plane = np.column_stack([pose_rotation[:, 0], pose_rotation[:, 1], pose_translation])
            homography = supersample @ matrix @ plane @ to_mm
            fine = cv2.warpPerspective(
                texture,
                homography,
                (2 * width, 2 * height),
                flags=cv2.INTER_LINEAR,
                borderValue=110,
            )
            image = cv2.resize(fine, (width, height), interpolation=cv2.INTER_AREA)
            noisy = np.clip(image + rng.normal(0, 1.5, image.shape), 0, 255).astype(np.uint8)
            ok, encoded = cv2.imencode(".png", noisy)
            if not ok:
                raise ValueError("PNG encoding failed")
            encoded.tofile(path / f"pair_{index:02d}.png")
    return truth
