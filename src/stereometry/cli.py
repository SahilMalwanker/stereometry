import argparse
from pathlib import Path

import cv2
import numpy as np

from .calibration import calibrate_folders, load_gray
from .report import depth_report, save_json
from .rig import StereoRig, triangulate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate, reconstruct and measure with stereo")
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("left", type=Path)
    calibration.add_argument("right", type=Path)
    calibration.add_argument("--columns", type=int, required=True)
    calibration.add_argument("--rows", type=int, required=True)
    calibration.add_argument("--square-mm", type=float, required=True)
    calibration.add_argument("--out", type=Path, default=Path("outputs/calibration"))
    depth = commands.add_parser("depth")
    depth.add_argument("left", type=Path)
    depth.add_argument("right", type=Path)
    depth.add_argument("--rig", type=Path, required=True)
    depth.add_argument("--num-disparities", type=int, default=64)
    depth.add_argument("--out", type=Path, default=Path("outputs/depth"))
    sparse = commands.add_parser("measure")
    sparse.add_argument("points", type=Path, help="NPZ with left_xy and right_xy raw pixels")
    sparse.add_argument("--rig", type=Path, required=True)
    sparse.add_argument("--out", type=Path, default=Path("outputs/measurement.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "calibrate":
            rig, report = calibrate_folders(
                args.left, args.right, args.columns, args.rows, args.square_mm
            )
            args.out.mkdir(parents=True, exist_ok=True)
            rig.save(args.out / "rig.json")
            save_json(args.out / "diagnostics.json", report)
            print(f"{report['views']} views; stereo RMS {report['stereo_rms_px']:.3f} px")
        elif args.command == "depth":
            _, report = depth_report(
                load_gray(args.left),
                load_gray(args.right),
                StereoRig.load(args.rig),
                args.out,
                num_disparities=args.num_disparities,
            )
            print(f"Valid pixels: {report['valid_fraction']:.1%}; outputs: {args.out}")
        else:
            with np.load(args.points, allow_pickle=False) as points:
                xyz, valid, errors = triangulate(
                    points["left_xy"], points["right_xy"], StereoRig.load(args.rig)
                )
            report = {
                "xyz_mm": [
                    point.tolist() if keep else None for point, keep in zip(xyz, valid, strict=True)
                ],
                "valid": valid.tolist(),
                "reprojection_error_px": [
                    float(error) if np.isfinite(error) else None for error in errors
                ],
                "distance_first_two_mm": float(np.linalg.norm(xyz[0] - xyz[1]))
                if len(xyz) >= 2 and valid[:2].all()
                else None,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            save_json(args.out, report)
            print(f"{int(valid.sum())}/{len(valid)} valid correspondences; outputs: {args.out}")
    except (OSError, ValueError, KeyError, cv2.error) as error:
        parser.error(str(error))
    return 0
