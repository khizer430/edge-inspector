import hashlib
import json
from pathlib import Path

import pytest

from src.data.synth_conveyor import run


@pytest.fixture
def small_cfg(tmp_path: Path) -> dict:
    return {
        "seed": 7,
        "fps": 10,
        "duration_s": 3,
        "frame_width": 320,
        "frame_height": 240,
        "raw_dir": str(tmp_path / "nonexistent"),
        "procedural_per_class": 5,
        "num_lanes": 2,
        "lane_margin_px": 20,
        "part_height_px": [40, 70],
        "rotation_deg": [-6, 6],
        "brightness": [0.9, 1.1],
        "defect_rate": 0.3,
        "belt_speed_px": [3.0, 5.0],
        "spawn_per_sec": 2.0,
        "ridge_spacing": 24,
        "belt_noise": 2,
        "min_visibility": 0.3,
        "post": {"gaussian_blur_sigma": 0.0, "motion_blur_px": 0, "brightness_scale": 1.0},
    }


def _hash_dir(p: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def test_determinism(tmp_path: Path, small_cfg: dict):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run(small_cfg, out_a)
    run(small_cfg, out_b)
    assert _hash_dir(out_a / "frames") == _hash_dir(out_b / "frames")
    la = json.loads((out_a / "labels.json").read_text())
    lb = json.loads((out_b / "labels.json").read_text())
    assert la["frames"] == lb["frames"]


def test_labels_are_well_formed(tmp_path: Path, small_cfg: dict):
    out = tmp_path / "run"
    run(small_cfg, out)
    data = json.loads((out / "labels.json").read_text())
    W, H = small_cfg["frame_width"], small_cfg["frame_height"]

    seen_ids = set()
    for fr in data["frames"]:
        for o in fr["objects"]:
            assert o["class"] in {"ok", "def"}
            x1, y1, x2, y2 = o["visible_bbox"]
            assert 0 <= x1 < x2 <= W
            assert 0 <= y1 < y2 <= H
            assert 0 < o["visibility"] <= 1.0
            seen_ids.add(o["track_id"])
    assert len(seen_ids) > 0


def test_yolo_labels_match_json(tmp_path: Path, small_cfg: dict):
    out = tmp_path / "run"
    run(small_cfg, out)
    data = json.loads((out / "labels.json").read_text())
    for fr in data["frames"]:
        yolo_path = out / "yolo" / "labels" / f"{fr['frame']:06d}.txt"
        lines = [l for l in yolo_path.read_text().splitlines() if l.strip()]
        assert len(lines) == len(fr["objects"])