"""Portable depth artifacts, preserving validity and acquisition identity."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .depth import anaglyph, dense_depth, write_ply


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_image(path, value):
    ok, encoded = cv2.imencode(".png", value)
    if not ok:
        raise ValueError("image encoding failed")
    encoded.tofile(path)


def depth_report(left, right, rig, out, *, synthetic=False, num_disparities=64):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result = dense_depth(left, right, rig, num_disparities=num_disparities)
    np.savez_compressed(
        out / "depth.npz",
        disparity_px=result.disparity_px,
        xyz_mm=result.xyz_mm,
        valid=result.valid,
        depth_sigma_mm=result.depth_sigma_mm,
    )
    count = write_ply(out / "cloud.ply", result)
    save_image(out / "left_rectified.png", result.left)
    save_image(out / "right_rectified.png", result.right)
    save_image(out / "valid.png", result.valid.astype(np.uint8) * 255)
    save_image(
        out / "anaglyph.png", cv2.cvtColor(anaglyph(result.left, result.right), cv2.COLOR_RGB2BGR)
    )
    rig.save(out / "rig.json")
    report = {
        "schema_version": 1,
        "synthetic": synthetic,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "length_unit": "mm",
        "ply_length_unit": "m",
        "left_pixels_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
        "right_pixels_sha256": hashlib.sha256(right.tobytes()).hexdigest(),
        "shape_hw": list(left.shape),
        "valid_fraction": float(result.valid.mean()),
        "ply_vertices": count,
        "num_disparities": num_disparities,
        "block_size": 5,
        "consistency_px": 1.5,
        "assumed_disparity_sigma_px": 0.25,
        "max_depth_mm": 10000,
        "median_depth_mm": float(np.median(result.xyz_mm[:, :, 2][result.valid]))
        if result.valid.any()
        else None,
    }
    save_json(out / "report.json", report)
    return result, report
