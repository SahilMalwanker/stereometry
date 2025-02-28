import numpy as np
import pytest

from stereometry.calibration import calibrate_folders, calibrate_points, checkerboard_points
from stereometry.cli import main
from stereometry.report import save_image
from stereometry.synthetic import calibration_observations, render_checkerboard_pairs


def test_known_rig_recovery():
    truth, objects, left, right = calibration_observations()
    recovered, diagnostics = calibrate_points(objects, left, right, truth.image_size)
    assert recovered.baseline_mm == pytest.approx(truth.baseline_mm, rel=0.01)
    assert recovered.left_matrix[0, 0] == pytest.approx(truth.left_matrix[0, 0], rel=0.02)
    assert diagnostics["stereo_rms_px"] < 0.3
    assert len(diagnostics["left_view_rms_px"]) == 18


def test_image_folders_detect_reject_and_recover(tmp_path):
    truth = render_checkerboard_pairs(tmp_path, pairs=9)
    blank = np.full((480, 640), 110, np.uint8)
    save_image(tmp_path / "left" / "pair_99.png", blank)
    save_image(tmp_path / "right" / "pair_99.png", blank)
    rig, diagnostics = calibrate_folders(tmp_path / "left", tmp_path / "right", 9, 6, 18)
    assert diagnostics["rejected_pairs"] == ["pair_99.png"]
    assert len(diagnostics["accepted_pairs"]) == 9
    assert rig.baseline_mm == pytest.approx(truth.baseline_mm, rel=0.02)
    assert rig.left_matrix[0, 0] == pytest.approx(truth.left_matrix[0, 0], rel=0.03)
    assert diagnostics["stereo_rms_px"] < 0.5


def test_calibrate_command_writes_rig(tmp_path):
    render_checkerboard_pairs(tmp_path / "captures", pairs=8, seed=5)
    out = tmp_path / "calibration"
    captures = tmp_path / "captures"
    args = ["calibrate", str(captures / "left"), str(captures / "right")]
    args += ["--columns", "9", "--rows", "6", "--square-mm", "18", "--out", str(out)]
    assert main(args) == 0
    assert (out / "rig.json").exists()
    assert (out / "diagnostics.json").exists()


def test_board_scale_and_bad_inputs(tmp_path):
    points = checkerboard_points(9, 6, 18)
    assert points.shape == (54, 3)
    assert np.max(points[:, 0]) == 144
    with pytest.raises(ValueError):
        checkerboard_points(9, 6, -12)
    with pytest.raises(ValueError):
        calibrate_points([], [], [], (640, 480))
    with pytest.raises(ValueError):
        calibrate_folders(tmp_path, tmp_path, 9, 6, 18)
