import cv2
import numpy as np
import pytest

from stereometry.rig import StereoRig, planar_points, split_side_by_side, triangulate


def rig():
    matrix = np.array([[400.0, 0, 240], [0, 400, 144], [0, 0, 1]])
    return StereoRig(matrix, np.zeros(5), matrix, np.zeros(5), np.eye(3), [-63, 0, 0], (480, 288))


def project(points, model):
    left = cv2.projectPoints(
        points, np.zeros(3), np.zeros(3), model.left_matrix, model.left_distortion
    )[0].reshape(-1, 2)
    right = cv2.projectPoints(
        points, np.zeros(3), model.translation_mm, model.right_matrix, model.right_distortion
    )[0].reshape(-1, 2)
    return left, right


def test_known_points_and_cheirality():
    model = rig()
    truth = np.array([[10.0, 20, 900], [50, -30, 1400], [0, 0, -900]])
    left, right = project(truth, model)
    reconstructed, valid, _ = triangulate(left, right, model)
    np.testing.assert_allclose(reconstructed[:2], truth[:2], atol=1e-6)
    assert valid.tolist() == [True, True, False]
    right[0, 1] += 30
    assert not triangulate(left, right, model)[1][0]


def test_planar_measurement_and_serialization(tmp_path):
    model = rig()
    board = np.array([[0.0, 0, 0], [20, 0, 0], [0, 20, 0], [20, 20, 0], [40, 20, 0], [40, 0, 0]])
    pixels = cv2.projectPoints(
        board,
        np.array([0.15, -0.2, 0.05]),
        np.array([0.0, 0, 800]),
        model.left_matrix,
        model.left_distortion,
    )[0].reshape(-1, 2)
    recovered = planar_points(pixels[:2], pixels, board, model.left_matrix, model.left_distortion)
    assert np.linalg.norm(recovered[0] - recovered[1]) == pytest.approx(20, abs=1e-5)
    model.save(tmp_path / "rig.json")
    assert StereoRig.load(tmp_path / "rig.json").baseline_mm == 63


def test_split_and_invalid_rig():
    frame = np.arange(80).reshape(8, 10)
    left, right = split_side_by_side(frame)
    np.testing.assert_array_equal(np.hstack([left, right]), frame)
    with pytest.raises(ValueError):
        split_side_by_side(frame[:, :9])
    model = rig()
    with pytest.raises(ValueError):
        StereoRig(
            model.left_matrix,
            np.zeros(5),
            model.right_matrix,
            np.zeros(5),
            np.eye(3),
            [0, 0, 0],
            (480, 288),
        )
