"""Reproduce a synthetic calibration and three-plane depth benchmark."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stereometry.calibration import calibrate_folders, calibrate_points
from stereometry.report import depth_report, save_image, save_json
from stereometry.synthetic import (
    calibration_observations,
    layered_scene,
    render_checkerboard_pairs,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "synthetic")
    parser.add_argument("--figures", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    truth, objects, left_points, right_points = calibration_observations()
    recovered, calibration = calibrate_points(objects, left_points, right_points, truth.image_size)
    recovered.save(args.out / "calibration_recovered.json")
    calibration.update(
        synthetic=True,
        known_baseline_mm=truth.baseline_mm,
        known_focal_x_px=float(truth.left_matrix[0, 0]),
        recovered_focal_x_px=float(recovered.left_matrix[0, 0]),
    )
    save_json(args.out / "calibration_diagnostics.json", calibration)
    captures = args.out / "captures"
    image_truth = render_checkerboard_pairs(captures, pairs=12)
    image_rig, image_calibration = calibrate_folders(
        captures / "left", captures / "right", 9, 6, 18
    )
    image_calibration.update(
        synthetic=True,
        known_baseline_mm=image_truth.baseline_mm,
        known_focal_x_px=float(image_truth.left_matrix[0, 0]),
        recovered_focal_x_px=float(image_rig.left_matrix[0, 0]),
    )
    rig, left, right, disparities = layered_scene()
    result, depth = depth_report(left, right, rig, args.out / "depth", synthetic=True)
    save_image(args.out / "left.png", left)
    save_image(args.out / "right.png", right)
    rig.save(args.out / "rectified_rig.json")
    depth["median_disparity_error_px"] = float(
        np.median(np.abs(result.disparity_px[result.valid] - disparities[result.valid]))
    )
    depth["planes"] = []
    for disparity in (18, 28, 42):
        mask = result.valid & (disparities == disparity)
        depth["planes"].append(
            {
                "disparity_px": disparity,
                "truth_mm": 400 * 63 / disparity,
                "median_mm": float(np.median(result.xyz_mm[:, :, 2][mask])),
                "median_sigma_mm": float(np.median(result.depth_sigma_mm[mask])),
            }
        )
    save_json(
        args.figures / "results.json",
        {"calibration": calibration, "image_calibration": image_calibration, "depth": depth},
    )
    figure = plt.figure(figsize=(13, 7), layout="constrained")
    grid = figure.add_gridspec(2, 3)
    panels = [
        ("Synthetic left", left, "gray"),
        ("Synthetic right", right, "gray"),
        ("Accepted correspondences", result.valid, "gray"),
        ("Disparity [pixels]", np.where(result.valid, result.disparity_px, np.nan), "viridis"),
        ("Depth sensitivity [mm], assumed sigma(d)=0.25 px", result.depth_sigma_mm, "magma"),
    ]
    for index, (title, image, cmap) in enumerate(panels):
        axis = figure.add_subplot(grid[index // 3, index % 3])
        artist = axis.imshow(image, cmap=cmap)
        axis.set_title(title, fontsize=10)
        axis.set_axis_off()
        if index >= 3:
            figure.colorbar(artist, ax=axis, shrink=0.7)
    axis = figure.add_subplot(grid[1, 2], projection="3d")
    mask = result.valid[::8, ::8]
    cloud = result.xyz_mm[::8, ::8][mask] / 1000
    axis.scatter(cloud[:, 0], cloud[:, 2], -cloud[:, 1], c=cloud[:, 2], cmap="viridis", s=2)
    axis.set(xlabel="X [m]", ylabel="Depth [m]", zlabel="Up [m]", title="Reconstructed planes")
    axis.view_init(elev=22, azim=-50)
    figure.savefig(args.figures / "overview.png", dpi=140)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    axes[0].plot(calibration["left_view_rms_px"], "o-", label="Left")
    axes[0].plot(calibration["right_view_rms_px"], "o-", label="Right")
    axes[0].set(
        xlabel="Synthetic view index", ylabel="RMS [px]", title="Per-view calibration residuals"
    )
    axes[0].legend()
    distances = np.linspace(400, 2500, 200)
    axes[1].plot(distances, distances**2 / (400 * 63) * 0.25, color="#087f73")
    axes[1].set(
        xlabel="Depth [mm]",
        ylabel="Propagated sigma(Z) [mm]",
        title="Depth sensitivity grows quadratically",
    )
    figure.savefig(args.figures / "diagnostics.png", dpi=140)
    plt.close(figure)
    print(f"Synthetic stereo RMS: {calibration['stereo_rms_px']:.4f} px")
    print(f"Valid dense pixels: {depth['valid_fraction']:.1%}")
    print(f"Median disparity error: {depth['median_disparity_error_px']:.4f} px")


if __name__ == "__main__":
    main()
