#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "openvla_rollout_graph_v2"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_episode_dirs(data_root: Path) -> list[Path]:
    return sorted(p.parent for p in (data_root / "episodes").glob("*/*/*/COMPLETE"))


def select_examples(episode_dirs: list[Path], per_group: int) -> list[Path]:
    groups: dict[tuple[str, bool], list[Path]] = defaultdict(list)
    for episode_dir in episode_dirs:
        meta_path = episode_dir / "metadata.json"
        frames_path = episode_dir / "frames.npz"
        if not meta_path.exists() or not frames_path.exists():
            continue
        meta = read_json(meta_path)
        key = (str(meta.get("policy_id")), bool(meta.get("episode_success")))
        if len(groups[key]) < per_group:
            groups[key].append(episode_dir)

    order = [
        ("high_official_libero_spatial", True),
        ("high_official_libero_spatial", False),
        ("low_10pct_action_only", True),
        ("low_10pct_action_only", False),
    ]
    selected: list[Path] = []
    for key in order:
        selected.extend(groups.get(key, []))
    return selected


def frame_indices(num_frames: int, count: int) -> list[int]:
    if num_frames <= 0:
        return []
    count = max(1, min(count, num_frames))
    return sorted(set(int(i) for i in np.linspace(0, num_frames - 1, num=count)))


def short_name(name: str) -> str:
    replacements = {
        "akita_black_bowl": "black_bowl",
        "glazed_rim_porcelain_ramekin": "ramekin",
        "wooden_cabinet": "cabinet",
        "flat_stove": "stove",
        "cookies": "cookies",
        "plate": "plate",
    }
    out = name
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def format_triplets(triplets: list[list[str]], max_triplets: int) -> list[str]:
    lines = []
    for s, p, o in triplets[:max_triplets]:
        lines.append(f"{short_name(s)}  {p}  {short_name(o)}")
    if len(triplets) > max_triplets:
        lines.append(f"... {len(triplets) - max_triplets} more")
    return lines


def draw_overlay(
    image: Image.Image,
    meta: dict[str, Any],
    frame_idx: int,
    max_triplets: int,
    position_record: dict[str, Any] | None = None,
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    frames = meta.get("frames", [])
    frame_meta = frames[frame_idx] if frame_idx < len(frames) else {}
    all_triplets = meta.get("oracle_graph_triplets", [])
    triplets = all_triplets[frame_idx] if frame_idx < len(all_triplets) else []

    image_positions = {}
    if position_record:
        image_positions = position_record.get("image_plane_positions", {}) or {}
    scale_x = image.width / 256.0
    scale_y = image.height / 256.0

    edge_colors = {
        "grasping": (255, 64, 64, 230),
        "on": (64, 210, 112, 210),
        "inside": (64, 150, 255, 210),
        "contains": (64, 150, 255, 210),
        "left_of": (255, 210, 64, 190),
        "right_of": (255, 210, 64, 190),
        "front_of": (220, 128, 255, 190),
        "behind": (220, 128, 255, 190),
        "above": (64, 230, 230, 190),
        "below": (64, 230, 230, 190),
    }
    anchored_edges = 0
    for s, pred, o in triplets[: max_triplets]:
        sp = image_positions.get(s)
        op = image_positions.get(o)
        if sp is None or op is None:
            continue
        x1, y1 = float(sp[0]) * scale_x, float(sp[1]) * scale_y
        x2, y2 = float(op[0]) * scale_x, float(op[1]) * scale_y
        color = edge_colors.get(pred, (255, 255, 255, 160))
        draw.line((x1, y1, x2, y2), fill=color, width=2)
        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        draw.rectangle((mx - 2, my - 2, mx + 2, my + 2), fill=color)
        anchored_edges += 1

    for i, (node_id, xy) in enumerate(sorted(image_positions.items())):
        x, y = float(xy[0]) * scale_x, float(xy[1]) * scale_y
        color = (255, 255, 255, 245)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 0, 0, 180), outline=color, width=2)
        label = short_name(node_id)
        draw.text((x + 7, y - 6), label, font=font, fill=color)

    header = [
        f"policy={meta.get('policy_id')}",
        f"task={meta.get('task_id')} success={meta.get('episode_success')} reason={meta.get('terminal_reason')}",
        f"t={frame_idx}/{max(0, int(meta.get('episode_length', 1)) - 1)} "
        f"grasping={frame_meta.get('has_grasping')} contacts={frame_meta.get('contact_count')} "
        f"triplets={len(triplets)} anchored_edges={anchored_edges}",
    ]
    relation_lines = format_triplets(triplets, max_triplets)
    text_lines = header + [""] + relation_lines

    wrapped: list[str] = []
    for line in text_lines:
        wrapped.extend(textwrap.wrap(line, width=54) if line else [""])

    pad = 8
    line_h = 13
    box_w = min(canvas.width - 12, 430)
    box_h = min(canvas.height - 12, pad * 2 + line_h * len(wrapped))
    draw.rectangle((6, 6, 6 + box_w, 6 + box_h), fill=(0, 0, 0, 178), outline=(255, 255, 255, 170))

    y = 6 + pad
    for i, line in enumerate(wrapped):
        if y > 6 + box_h - line_h:
            break
        fill = (255, 230, 128, 255) if i < len(header) else (255, 255, 255, 255)
        draw.text((6 + pad, y), line, font=font, fill=fill)
        y += line_h

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def load_position_records(episode_dir: Path) -> dict[int, dict[str, Any]]:
    path = episode_dir / "graph3d_positions.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    records = payload.get("position_records", [])
    return {int(r["timestep"]): r for r in records if "timestep" in r}


def make_contact_sheet(frames: list[Image.Image], columns: int = 3) -> Image.Image:
    if not frames:
        raise ValueError("no frames")
    w, h = frames[0].size
    columns = max(1, min(columns, len(frames)))
    rows = int(np.ceil(len(frames) / columns))
    sheet = Image.new("RGB", (w * columns, h * rows), (255, 255, 255))
    for i, frame in enumerate(frames):
        sheet.paste(frame, ((i % columns) * w, (i // columns) * h))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay rollout oracle scene-graph triplets on RGB frames.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--per-group", type=int, default=1)
    parser.add_argument("--frames-per-episode", type=int, default=6)
    parser.add_argument("--max-triplets", type=int, default=18)
    parser.add_argument("--episode-dir", type=Path, action="append", default=[])
    args = parser.parse_args()

    out_dir = args.output_dir or (args.data_root / "inspection" / f"graph_overlay_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = args.episode_dir or select_examples(iter_episode_dirs(args.data_root), args.per_group)
    written: list[str] = []
    summary: list[dict[str, Any]] = []

    for episode_dir in episode_dirs:
        meta = read_json(episode_dir / "metadata.json")
        data = np.load(episode_dir / "frames.npz")
        rgbs = data["rgb"]
        chosen = frame_indices(int(rgbs.shape[0]), args.frames_per_episode)
        positions = load_position_records(episode_dir)
        panels = []
        for idx in chosen:
            panels.append(draw_overlay(Image.fromarray(rgbs[idx]), meta, idx, args.max_triplets, positions.get(idx)))
        sheet = make_contact_sheet(panels)
        stem = f"{meta.get('policy_id')}_success-{meta.get('episode_success')}_task-{meta.get('task_id')}_{meta.get('episode_id')}"
        out_path = out_dir / f"{stem}.jpg"
        sheet.save(out_path, quality=92)
        written.append(str(out_path))
        summary.append(
            {
                "episode_dir": str(episode_dir),
                "output_path": str(out_path),
                "policy_id": meta.get("policy_id"),
                "task_id": meta.get("task_id"),
                "episode_success": meta.get("episode_success"),
                "episode_length": meta.get("episode_length"),
                "frames": chosen,
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(written)} overlay sheets to {out_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
