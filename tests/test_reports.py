import json

from stereometry.cli import main
from stereometry.report import save_image
from stereometry.synthetic import layered_scene


def test_dense_cli(tmp_path):
    rig, left, right, _ = layered_scene()
    rig.save(tmp_path / "rig.json")
    save_image(tmp_path / "left.png", left)
    save_image(tmp_path / "right.png", right)
    out = tmp_path / "report"
    assert (
        main(
            [
                "depth",
                str(tmp_path / "left.png"),
                str(tmp_path / "right.png"),
                "--rig",
                str(tmp_path / "rig.json"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads((out / "report.json").read_text())
    assert report["valid_fraction"] > 0.5
    assert report["length_unit"] == "mm"
    assert (out / "depth.npz").exists()
    assert (out / "cloud.ply").exists()
