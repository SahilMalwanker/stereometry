"""Dense reconstruction with correspondence and image-support validity masks."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .rig import StereoRig


@dataclass
class DepthResult:
    left: np.ndarray
    right: np.ndarray
    disparity_px: np.ndarray
    xyz_mm: np.ndarray
    valid: np.ndarray
    depth_sigma_mm: np.ndarray


def dense_depth(
    left,
    right,
    rig: StereoRig,
    *,
    num_disparities=64,
    block_size=5,
    consistency_px=1.5,
    disparity_sigma_px=0.25,
    max_depth_mm=10000.0,
):
    for image in (left, right):
        if image.dtype != np.uint8 or image.ndim != 2 or image.shape[::-1] != rig.image_size:
            raise ValueError("images must be uint8 grayscale at the calibrated resolution")
    if num_disparities < 16 or num_disparities % 16 or num_disparities >= left.shape[1] - 16:
        raise ValueError(
            "num_disparities must be a multiple of 16 smaller than image width minus 16"
        )
    if block_size < 3 or block_size > 21 or block_size % 2 == 0:
        raise ValueError("block_size must be odd, between 3 and 21")
    if not all(
        np.isfinite(value) and value > 0
        for value in (consistency_px, disparity_sigma_px, max_depth_mm)
    ):
        raise ValueError(
            "consistency, disparity sigma and depth limits must be finite and positive"
        )
    rotations = cv2.stereoRectify(
        rig.left_matrix,
        rig.left_distortion,
        rig.right_matrix,
        rig.right_distortion,
        rig.image_size,
        rig.rotation,
        rig.translation_mm.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    left_rotation, right_rotation, left_projection, right_projection, reprojection = rotations[:5]
    if right_projection[0, 3] >= 0 or abs(right_projection[1, 3]) > 1e-6:
        raise ValueError(
            "dense matching requires a horizontal rig with the right camera to the right"
        )
    rectified, supports = [], []
    for image, matrix, distortion, rotation, projection in (
        (left, rig.left_matrix, rig.left_distortion, left_rotation, left_projection),
        (right, rig.right_matrix, rig.right_distortion, right_rotation, right_projection),
    ):
        map_x, map_y = cv2.initUndistortRectifyMap(
            matrix, distortion, rotation, projection, rig.image_size, cv2.CV_32FC1
        )
        rectified.append(cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR))
        supports.append(
            (map_x >= 1)
            & (map_x < image.shape[1] - 2)
            & (map_y >= 1)
            & (map_y < image.shape[0] - 2)
        )

    def matcher(min_disparity):
        return cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * block_size**2,
            P2=32 * block_size**2,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2,
            disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    left_rectified, right_rectified = rectified
    disparity = matcher(0).compute(left_rectified, right_rectified).astype(np.float32) / 16
    reverse = (
        matcher(-num_disparities).compute(right_rectified, left_rectified).astype(np.float32) / 16
    )
    rows, columns = np.indices(left.shape)
    right_column = np.rint(columns - disparity).astype(int)
    in_bounds = (right_column >= 0) & (right_column < left.shape[1])
    safe_column = np.clip(right_column, 0, left.shape[1] - 1)
    reverse_sample = reverse[rows, safe_column]
    valid = (
        (disparity > 0)
        & in_bounds
        & supports[0]
        & supports[1][rows, safe_column]
        & (reverse_sample > -num_disparities - 1)
        & (reverse_sample < 0)
        & (np.abs(disparity + reverse_sample) <= consistency_px)
    )
    intensity = left_rectified.astype(np.float32)
    mean = cv2.boxFilter(intensity, -1, (9, 9))
    variance = cv2.boxFilter(intensity**2, -1, (9, 9)) - mean**2
    valid &= variance > 9
    xyz = cv2.reprojectImageTo3D(disparity, reprojection)
    valid &= np.isfinite(xyz).all(axis=2) & (xyz[:, :, 2] > 0) & (xyz[:, :, 2] <= max_depth_mm)
    xyz[~valid] = np.nan
    sigma = np.full(disparity.shape, np.nan, np.float32)
    sigma[valid] = abs(right_projection[0, 3]) * disparity_sigma_px / disparity[valid] ** 2
    return DepthResult(left_rectified, right_rectified, disparity, xyz, valid, sigma)


def write_ply(path, result: DepthResult, stride=2):
    """Export valid points in meters; no invalid placeholders are written."""
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    mask = result.valid[::stride, ::stride]
    points = result.xyz_mm[::stride, ::stride][mask] / 1000
    values = result.left[::stride, ::stride][mask]
    with Path(path).open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\ncomment length_unit meter\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, value in zip(points, values, strict=True):
            stream.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {value} {value} {value}\n")
    return len(points)


def anaglyph(left, right):
    return np.dstack([left, right, right])


def epipolar_lines(left_xy, rig: StereoRig):
    """Lines in undistorted right pixels from raw left pixels; each line has unit normal."""
    skew = np.array(
        [
            [0, -rig.translation_mm[2], rig.translation_mm[1]],
            [rig.translation_mm[2], 0, -rig.translation_mm[0]],
            [-rig.translation_mm[1], rig.translation_mm[0], 0],
        ]
    )
    fundamental = (
        np.linalg.inv(rig.right_matrix).T @ skew @ rig.rotation @ np.linalg.inv(rig.left_matrix)
    )
    pixels = cv2.undistortPoints(
        np.asarray(left_xy, dtype=float).reshape(-1, 1, 2),
        rig.left_matrix,
        rig.left_distortion,
        P=rig.left_matrix,
    ).reshape(-1, 2)
    lines = np.column_stack([pixels, np.ones(len(pixels))]) @ fundamental.T
    normals = np.linalg.norm(lines[:, :2], axis=1)
    if np.any(normals < 1e-12):
        raise ValueError("epipolar line undefined at the epipole")
    return lines / normals[:, None]
