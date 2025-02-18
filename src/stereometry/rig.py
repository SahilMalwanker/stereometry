"""OpenCV column-vector geometry; world coordinates are left-camera millimeters."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class StereoRig:
    left_matrix: np.ndarray
    left_distortion: np.ndarray
    right_matrix: np.ndarray
    right_distortion: np.ndarray
    rotation: np.ndarray
    translation_mm: np.ndarray
    image_size: tuple[int, int]

    def __post_init__(self):
        for name in ("left_matrix", "right_matrix", "rotation"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3, 3) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite 3x3 matrix")
            setattr(self, name, value)
        for name in ("left_distortion", "right_distortion"):
            value = np.asarray(getattr(self, name), dtype=float).ravel()
            if value.shape != (5,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain k1,k2,p1,p2,k3")
            setattr(self, name, value)
        self.translation_mm = np.asarray(self.translation_mm, dtype=float).reshape(3)
        if not np.isfinite(self.translation_mm).all() or self.baseline_mm <= 0:
            raise ValueError("a finite nonzero baseline is required")
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-5):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(self.rotation), 1, atol=1e-5):
            raise ValueError("rotation must have determinant +1")
        for matrix in (self.left_matrix, self.right_matrix):
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
                raise ValueError("invalid intrinsic matrix")
        if len(self.image_size) != 2 or any(
            int(value) != value or value < 16 for value in self.image_size
        ):
            raise ValueError("image_size must be (width,height), each at least 16 pixels")
        self.image_size = tuple(map(int, self.image_size))

    @property
    def baseline_mm(self):
        return float(np.linalg.norm(self.translation_mm))

    def save(self, path):
        data = {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in vars(self).items()
        }
        data.update(schema_version=1, length_unit="mm")
        Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.pop("schema_version", None) != 1 or data.pop("length_unit", None) != "mm":
            raise ValueError("expected schema_version 1 and length_unit mm")
        return cls(**data)


def triangulate(left_xy, right_xy, rig: StereoRig, max_error_px=2.0):
    """Raw distorted pixel correspondences -> XYZ, validity, max reprojection error."""
    left = np.asarray(left_xy, dtype=float)
    right = np.asarray(right_xy, dtype=float)
    if left.ndim != 2 or left.shape[1] != 2 or right.shape != left.shape or not len(left):
        raise ValueError("left/right points must be equal nonempty Nx2 arrays")
    if not np.isfinite(left).all() or not np.isfinite(right).all() or max_error_px <= 0:
        raise ValueError("points must be finite and error limit positive")
    normalized_left = cv2.undistortPoints(
        left.reshape(-1, 1, 2), rig.left_matrix, rig.left_distortion
    ).reshape(-1, 2)
    normalized_right = cv2.undistortPoints(
        right.reshape(-1, 1, 2), rig.right_matrix, rig.right_distortion
    ).reshape(-1, 2)
    left_projection = np.column_stack([np.eye(3), np.zeros(3)])
    right_projection = np.column_stack([rig.rotation, rig.translation_mm])
    homogeneous = cv2.triangulatePoints(
        left_projection, right_projection, normalized_left.T, normalized_right.T
    ).T
    with np.errstate(divide="ignore", invalid="ignore"):
        xyz = homogeneous[:, :3] / homogeneous[:, 3:]
    right_xyz = xyz @ rig.rotation.T + rig.translation_mm
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0) & (right_xyz[:, 2] > 0)
    errors = np.full(len(xyz), np.inf)
    if valid.any():
        projection_left, _ = cv2.projectPoints(
            xyz[valid], np.zeros(3), np.zeros(3), rig.left_matrix, rig.left_distortion
        )
        projection_right, _ = cv2.projectPoints(
            xyz[valid],
            cv2.Rodrigues(rig.rotation)[0],
            rig.translation_mm,
            rig.right_matrix,
            rig.right_distortion,
        )
        errors[valid] = np.maximum(
            np.linalg.norm(projection_left.reshape(-1, 2) - left[valid], axis=1),
            np.linalg.norm(projection_right.reshape(-1, 2) - right[valid], axis=1),
        )
    valid &= errors <= max_error_px
    xyz[~valid] = np.nan
    return xyz, valid, errors


def planar_points(image_xy, board_xy, board_world, matrix, distortion):
    """Raw pixels -> board-plane XY in the length units of board_world."""
    world = np.asarray(board_world, dtype=float)
    if world.ndim != 2 or world.shape[1] != 3 or len(world) < 4 or not np.allclose(world[:, 2], 0):
        raise ValueError("board_world must contain at least four points on Z=0")
    detected = np.asarray(board_xy, dtype=float).reshape(-1, 2)
    if len(detected) != len(world):
        raise ValueError("board pixel and world point counts differ")
    ok, rotation, translation = cv2.solvePnP(world, detected, matrix, distortion)
    if not ok:
        raise ValueError("board pose estimation failed")
    rays = cv2.undistortPoints(
        np.asarray(image_xy, dtype=float).reshape(-1, 1, 2), matrix, distortion
    ).reshape(-1, 2)
    rotation_matrix = cv2.Rodrigues(rotation)[0]
    origin = -rotation_matrix.T @ translation.ravel()
    directions = np.column_stack([rays, np.ones(len(rays))]) @ rotation_matrix
    if np.any(np.abs(directions[:, 2]) < 1e-10):
        raise ValueError("measurement ray is parallel to the board")
    scales = -origin[2] / directions[:, 2]
    if np.any(scales <= 0):
        raise ValueError("measurement ray points away from the board")
    return (origin + scales[:, None] * directions)[:, :2]


def split_side_by_side(frame):
    frame = np.asarray(frame)
    if frame.ndim not in (2, 3) or not frame.size or frame.shape[1] % 2:
        raise ValueError("expected a nonempty side-by-side frame with even width")
    midpoint = frame.shape[1] // 2
    return frame[:, :midpoint].copy(), frame[:, midpoint:].copy()
