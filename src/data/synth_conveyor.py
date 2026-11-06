"""Synthetic conveyor generator with perfect detection/tracking GT."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class PartSprite:
    image: np.ndarray
    label: str
    source: str


def procedural_sprite(rng: np.random.Generator, label: str, size: int = 160) -> np.ndarray:
    img = np.zeros((size, size, 4), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    radius = int(size * 0.42)
    gray = int(rng.integers(120, 180))

    if rng.random() < 0.5:
        cv2.circle(img, (cx, cy), radius, (gray, gray, gray, 255), -1)
    else:
        pts = cv2.ellipse2Poly((cx, cy), (radius, radius), 0, 0, 360, 6)
        cv2.fillPoly(img, [pts], (gray, gray, gray, 255))

    for r in range(radius // 5, radius, max(4, radius // 8)):
        shade = int(np.clip(gray + rng.integers(-15, 15), 0, 255))
        cv2.circle(img, (cx, cy), r, (shade, shade, shade, 255), 1)
    cv2.circle(img, (cx, cy), max(3, radius // 6), (40, 40, 40, 255), -1)

    noise = rng.normal(0, 6, (size, size, 3)).astype(np.int16)
    rgb = np.clip(img[..., :3].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[..., :3] = rgb

    if label == "def":
        for _ in range(int(rng.integers(1, 3))):
            x0 = int(rng.integers(cx - radius, cx + radius))
            y0 = int(rng.integers(cy - radius, cy + radius))
            pts = np.array([[x0, y0]], dtype=np.int32)
            for _ in range(int(rng.integers(3, 7))):
                x0 += int(rng.integers(-15, 15))
                y0 += int(rng.integers(-15, 15))
                pts = np.vstack([pts, [x0, y0]])
            cv2.polylines(img, [pts], False, (30, 30, 30, 255), 2)
        dx = int(rng.integers(cx - radius // 2, cx + radius // 2))
        dy = int(rng.integers(cy - radius // 2, cy + radius // 2))
        cv2.circle(img, (dx, dy), int(rng.integers(4, 10)), (60, 60, 60, 255), -1)

    return img


def load_sprites(raw_dir: Path, rng: np.random.Generator,
                 procedural_per_class: int) -> dict[str, list[PartSprite]]:
    sprites: dict[str, list[PartSprite]] = {"ok": [], "def": []}
    mapping = {"ok_front": "ok", "ok": "ok", "def_front": "def", "def": "def"}

    if raw_dir.exists():
        for sub in sorted(raw_dir.iterdir()):
            if not sub.is_dir():
                continue
            label = mapping.get(sub.name.lower())
            if label is None:
                continue
            for p in sorted(sub.iterdir()):
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if im is None:
                    continue
                if im.ndim == 2:
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
                elif im.shape[2] == 3:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
                sprites[label].append(PartSprite(im, label, str(p)))

    for label in ("ok", "def"):
        for _ in range(procedural_per_class):
            sprites[label].append(PartSprite(procedural_sprite(rng, label), label, "procedural"))

    if not sprites["ok"] or not sprites["def"]:
        raise RuntimeError("No sprites available for one of the classes.")
    return sprites


def make_belt(width: int, height: int) -> np.ndarray:
    belt = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        v = int(55 + 30 * (1 - abs(y - height / 2) / (height / 2)))
        belt[y, :] = (v, v, v)
    return belt


@dataclass
class SpawnedPart:
    track_id: int
    sprite: PartSprite
    x: float
    y: float
    scale: float
    angle: float
    brightness: float


def _render_sprite(p: SpawnedPart) -> np.ndarray:
    src = p.sprite.image
    new_w = max(1, int(src.shape[1] * p.scale))
    new_h = max(1, int(src.shape[0] * p.scale))
    resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)

    rgb = np.clip(resized[..., :3].astype(np.float32) * p.brightness, 0, 255).astype(np.uint8)
    resized[..., :3] = rgb

    M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), p.angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    rot_w = int(new_h * sin + new_w * cos)
    rot_h = int(new_h * cos + new_w * sin)
    M[0, 2] += rot_w / 2 - new_w / 2
    M[1, 2] += rot_h / 2 - new_h / 2
    return cv2.warpAffine(
        resized, M, (rot_w, rot_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def compose_frame(base_belt: np.ndarray, active: list[SpawnedPart],
                  cfg: dict, rng: np.random.Generator,
                  belt_offset: float) -> tuple[np.ndarray, list[dict]]:
    H, W = base_belt.shape[:2]
    frame = base_belt.copy()

    spacing = cfg.get("ridge_spacing", 48)
    off = int(belt_offset) % spacing
    for x in range(-spacing, W + spacing, spacing):
        cv2.line(frame, (x + off, 0), (x + off, H), (42, 42, 42), 2)

    noise = rng.normal(0, cfg.get("belt_noise", 3), frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    min_vis = cfg.get("min_visibility", 0.30)
    labels: list[dict] = []

    for p in active:
        rot = _render_sprite(p)
        rh, rw = rot.shape[:2]
        x1, y1 = int(round(p.x)), int(round(p.y))
        x2, y2 = x1 + rw, y1 + rh

        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(W, x2), min(H, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        visible_area = (cx2 - cx1) * (cy2 - cy1)
        full_area = max(1, rw * rh)
        if visible_area / full_area < min_vis:
            continue

        sx1, sy1 = cx1 - x1, cy1 - y1
        sx2, sy2 = sx1 + (cx2 - cx1), sy1 + (cy2 - cy1)
        patch = rot[sy1:sy2, sx1:sx2]
        alpha = patch[..., 3:4].astype(np.float32) / 255.0
        roi = frame[cy1:cy2, cx1:cx2].astype(np.float32)
        comp = patch[..., :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
        frame[cy1:cy2, cx1:cx2] = comp.astype(np.uint8)

        labels.append({
            "track_id": p.track_id,
            "class": p.sprite.label,
            "class_id": 0 if p.sprite.label == "ok" else 1,
            "full_bbox": [x1, y1, x2, y2],
            "visible_bbox": [cx1, cy1, cx2, cy2],
            "visibility": round(visible_area / full_area, 4),
            "angle": round(p.angle, 3),
            "scale": round(p.scale, 4),
        })

    return frame, labels


def run(cfg: dict, out_dir: Path) -> None:
    rng = np.random.default_rng(cfg["seed"])
    py_rng = random.Random(cfg["seed"])

    sprites = load_sprites(Path(cfg["raw_dir"]), rng, cfg.get("procedural_per_class", 40))

    W, H = cfg["frame_width"], cfg["frame_height"]
    base_belt = make_belt(W, H)
    fps, dur = cfg["fps"], cfg["duration_s"]
    total_frames = int(fps * dur)

    lane_count = cfg.get("num_lanes", 3)
    margin = cfg.get("lane_margin_px", 40)
    lane_h = (H - 2 * margin) / lane_count
    spawn_per_sec = cfg.get("spawn_per_sec", 1.6)
    defect_rate = cfg.get("defect_rate", 0.2)
    speed_lo, speed_hi = cfg["belt_speed_px"]

    frame_dir = out_dir / "frames"
    yolo_dir = out_dir / "yolo" / "labels"
    frame_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    preview_path = out_dir / "preview.mp4"
    writer = cv2.VideoWriter(str(preview_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        preview_path = out_dir / "preview.avi"
        writer = cv2.VideoWriter(str(preview_path),
                                 cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))

    active: list[SpawnedPart] = []
    next_id = 0
    spawn_acc = 0.0
    belt_offset = 0.0
    all_frames: list[dict] = []

    for t in range(total_frames):
        spawn_acc += spawn_per_sec / fps
        while spawn_acc >= 1.0:
            spawn_acc -= 1.0
            label = "def" if py_rng.random() < defect_rate else "ok"
            sprite = sprites[label][py_rng.randrange(len(sprites[label]))]
            target_h = float(rng.uniform(*cfg["part_height_px"]))
            scale = target_h / sprite.image.shape[0]
            lane = py_rng.randrange(lane_count)
            lane_y = margin + lane * lane_h + float(rng.uniform(0, max(0, lane_h - target_h)))
            active.append(SpawnedPart(
                track_id=next_id,
                sprite=sprite,
                x=-sprite.image.shape[1] * scale,
                y=lane_y,
                scale=scale,
                angle=float(rng.uniform(*cfg["rotation_deg"])),
                brightness=float(rng.uniform(*cfg["brightness"])),
            ))
            next_id += 1

        step = float(rng.uniform(speed_lo, speed_hi))
        for p in active:
            p.x += step
        belt_offset += step
        active = [p for p in active if p.x < W + 20]

        frame, labels = compose_frame(base_belt, active, cfg, rng, belt_offset)

        post = cfg.get("post") or {}
        if post.get("gaussian_blur_sigma", 0) > 0:
            frame = cv2.GaussianBlur(frame, (0, 0), float(post["gaussian_blur_sigma"]))
        if post.get("motion_blur_px", 0) > 0:
            k = int(post["motion_blur_px"])
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            frame = cv2.filter2D(frame, -1, kernel)
        if post.get("brightness_scale", 1.0) != 1.0:
            frame = np.clip(frame.astype(np.float32) * float(post["brightness_scale"]), 0, 255).astype(np.uint8)

        cv2.imwrite(str(frame_dir / f"{t:06d}.jpg"), frame)
        writer.write(frame)
        if t == 0:
            cv2.imwrite(str(out_dir / "preview_first.png"), frame)

        all_frames.append({"frame": t, "objects": labels})

        with open(yolo_dir / f"{t:06d}.txt", "w") as f:
            for o in labels:
                x1, y1, x2, y2 = o["visible_bbox"]
                cx = (x1 + x2) / 2 / W
                cy = (y1 + y2) / 2 / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                f.write(f"{o['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    writer.release()

    with open(out_dir / "labels.json", "w") as f:
        json.dump({"config": cfg, "frames": all_frames}, f, indent=1)

    print(f"[synth] {total_frames} frames → {out_dir}")
    print(f"[synth] preview → {preview_path}")
    print(f"[synth] total unique tracks: {next_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    args.out.mkdir(parents=True, exist_ok=True)
    run(cfg, args.out)


if __name__ == "__main__":
    main()