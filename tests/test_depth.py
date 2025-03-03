import numpy as np
import pytest

from stereometry.depth import dense_depth, epipolar_lines, write_ply
from stereometry.synthetic import layered_scene


def test_dense_planes_and_uncertainty(tmp_path):
    rig, left, right, truth = layered_scene()
    result = dense_depth(left, right, rig)
    assert result.valid.mean() > 0.5
    assert np.median(np.abs(result.disparity_px[result.valid] - truth[result.valid])) < 0.5
    for disparity in (18, 28, 42):
        mask = result.valid & (truth == disparity)
        assert mask.sum() > 1000
        expected = 400 * 63 / disparity
        assert np.median(result.xyz_mm[:, :, 2][mask]) == pytest.approx(expected, rel=0.03)
    assert np.isnan(result.xyz_mm[~result.valid]).all()
    assert np.nanmin(result.depth_sigma_mm) > 0
    count = write_ply(tmp_path / "cloud.ply", result, stride=8)
    assert f"element vertex {count}" in (tmp_path / "cloud.ply").read_text()


def test_blank_frames_have_no_valid_depth():
    rig, left, _, _ = layered_scene()
    blank = np.zeros_like(left)
    assert not dense_depth(blank, blank, rig).valid.any()


def test_epipolar_line_and_invalid_parameters():
    rig, left, right, _ = layered_scene()
    lines = epipolar_lines([[200, 90]], rig)
    assert abs(lines[0] @ [150, 90, 1]) < 1e-9
    with pytest.raises(ValueError):
        dense_depth(left, right, rig, num_disparities=30)
    with pytest.raises(ValueError):
        dense_depth(left, right, rig, disparity_sigma_px=-1)
