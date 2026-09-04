#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "openvla_rollout_graph_v2"

COLORS = {
    "left_of": (40, 125, 255),
    "right_of": (40, 125, 255),
    "above": (34, 150, 84),
    "below": (34, 150, 84),
    "front_of": (0, 150, 140),
    "behind": (0, 150, 140),
    "on": (235, 126, 35),
    "inside": (132, 86, 190),
    "contains": (132, 86, 190),
    "grasping": (220, 55, 90),
}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_episode_dirs(data_root: Path) -> list[Path]:
    return sorted(p.parent for p in (data_root / "episodes").glob("*/*/*/COMPLETE"))


def select_examples(episode_dirs: list[Path], per_group: int) -> list[Path]:
    groups: dict[tuple[str, bool], list[Path]] = defaultdict(list)
    for episode_dir in episode_dirs:
        meta_path = episode_dir / "metadata.json"
        if not meta_path.exists():
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


def load_position_records(episode_dir: Path) -> dict[int, dict[str, Any]]:
    path = episode_dir / "graph3d_positions.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run backfill_rollout_xyz_targets.py for this episode first")
    payload = read_json(path)
    records = payload.get("position_records", [])
    return {int(r["timestep"]): r for r in records if "timestep" in r}


def short_name(name: str) -> str:
    replacements = {
        "akita_black_bowl": "black_bowl",
        "glazed_rim_porcelain_ramekin": "ramekin",
        "wooden_cabinet": "cabinet",
        "flat_stove": "stove",
        "cookies": "cookies",
        "plate": "plate",
        "_1_burner_plate": "_burner",
        "_1_cabinet_top": "_top",
        "_1_button": "_button",
    }
    out = str(name)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def scaled_positions(raw: dict[str, Any], width: int, height: int, source_size: int = 256) -> dict[str, tuple[float, float]]:
    sx = width / float(source_size)
    sy = height / float(source_size)
    out = {}
    for node_id, xy in raw.items():
        if xy is None or len(xy) < 2:
            continue
        out[str(node_id)] = (float(xy[0]) * sx, float(xy[1]) * sy)
    return out


def scaled_anchors(record: dict[str, Any], width: int, height: int, source_size: int = 256) -> dict[str, dict[str, Any]]:
    raw_anchors = record.get("image_anchors", {}) or {}
    if raw_anchors:
        sx = width / float(source_size)
        sy = height / float(source_size)
        out: dict[str, dict[str, Any]] = {}
        for node_id, payload in raw_anchors.items():
            xy = payload.get("xy")
            if xy is None or len(xy) < 2:
                continue
            out[str(node_id)] = {
                **payload,
                "point": (float(xy[0]) * sx, float(xy[1]) * sy),
            }
        return out
    return {
        node_id: {"point": point, "source": "image_plane_positions", "visible": True, "visible_pixels": 0}
        for node_id, point in scaled_positions(record.get("image_plane_positions", {}) or {}, width, height, source_size).items()
    }


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int, int],
    width: int,
) -> tuple[float, float] | None:
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length < 1.0:
        return None
    pad = 10.0
    sx += dx / length * pad
    sy += dy / length * pad
    ex -= dx / length * pad
    ey -= dy / length * pad
    draw.line((sx, sy, ex, ey), fill=color, width=width)
    angle = math.atan2(ey - sy, ex - sx)
    head = 6
    for offset in (2.55, -2.55):
        hx = ex + head * math.cos(angle + offset)
        hy = ey + head * math.sin(angle + offset)
        draw.line((ex, ey, hx, hy), fill=color, width=width)
    return (sx + ex) * 0.5, (sy + ey) * 0.5


def draw_label_box(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill: tuple[int, int, int, int]) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text)
    pad = 2
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=fill)
    draw.text((x, y), text, fill=(0, 0, 0, 255))


def draw_frame(
    rgb: np.ndarray,
    meta: dict[str, Any],
    frame_idx: int,
    position_record: dict[str, Any],
    *,
    predicates: set[str] | None,
    max_edges: int | None,
    label_edges: bool,
    node_labels: bool,
    draw_title: bool,
    draw_legend: bool,
) -> Image.Image:
    image = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    anchors = scaled_anchors(position_record, image.width, image.height)
    image_positions = {node_id: payload["point"] for node_id, payload in anchors.items()}
    triplets = meta.get("oracle_graph_triplets", [])[frame_idx]
    if predicates:
        triplets = [t for t in triplets if t[1] in predicates]
    if max_edges is not None:
        priority = {"grasping": 0, "on": 1, "inside": 2, "contains": 2, "above": 3, "below": 3, "front_of": 4, "behind": 4, "left_of": 5, "right_of": 5}
        triplets = sorted(triplets, key=lambda t: (priority.get(t[1], 9), t[0], t[2]))[:max_edges]

    counts = Counter(t[1] for t in triplets)
    anchored_edges = 0
    for subject, predicate, obj in triplets:
        if subject not in image_positions or obj not in image_positions:
            continue
        color3 = COLORS.get(predicate, (80, 80, 80))
        alpha = 235 if predicate in {"grasping", "on", "inside", "contains"} else 145
        width = 4 if predicate == "grasping" else 2
        midpoint = draw_arrow(draw, image_positions[subject], image_positions[obj], (*color3, alpha), width)
        if midpoint is not None:
            anchored_edges += 1
            if label_edges and predicate in {"grasping", "on", "inside", "contains"}:
                draw.text((midpoint[0] + 3, midpoint[1] + 3), predicate, fill=(*color3, 255), font=font)

    node_ids = sorted(image_positions)
    for idx, node_id in enumerate(node_ids, 1):
        x, y = image_positions[node_id]
        radius = 6 if node_id != "gripper" else 7
        source = str(anchors[node_id].get("source", ""))
        if source == "world_projection":
            fill = (255, 242, 170, 190)
            outline = (150, 80, 20, 245)
        else:
            fill = (20, 20, 20, 230)
            outline = (255, 255, 255, 250)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
        if node_labels:
            label = short_name(node_id)
            lx = min(max(x + 8, 0), image.width - 55)
            ly = min(max(y - 7, 0), image.height - 12)
            draw_label_box(draw, (lx, ly), label, (255, 255, 255, 210))

    if draw_title:
        draw.rectangle((0, 0, image.width, 22), fill=(255, 255, 255, 210))
        title = (
            f"task={meta.get('task_id')} t={frame_idx} success={meta.get('episode_success')} "
            f"nodes={len(node_ids)} edges={anchored_edges}/{len(triplets)}"
        )
        draw.text((5, 5), title, fill=(0, 0, 0, 255), font=font)

    if draw_legend:
        legend_x = 5
        legend_y = image.height - 12 * (len(counts) + 1) - 4
        draw.rectangle((0, legend_y - 4, 150, image.height), fill=(255, 255, 255, 190))
        for pred in sorted(counts):
            color3 = COLORS.get(pred, (80, 80, 80))
            draw.line((legend_x, legend_y + 6, legend_x + 22, legend_y + 6), fill=(*color3, 255), width=3)
            draw.text((legend_x + 28, legend_y), f"{pred}: {counts[pred]}", fill=(0, 0, 0, 255), font=font)
            legend_y += 12

    return image.convert("RGB")


def make_sheet(images: list[Image.Image], columns: int) -> Image.Image:
    if not images:
        raise ValueError("no images")
    w, h = images[0].size
    columns = max(1, min(columns, len(images)))
    rows = int(math.ceil(len(images) / columns))
    sheet = Image.new("RGB", (w * columns, h * rows), "white")
    for i, image in enumerate(images):
        sheet.paste(image, ((i % columns) * w, (i // columns) * h))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw node-edge scene graph overlays on rollout RGB frames.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episode-dir", type=Path, action="append", default=[])
    parser.add_argument("--per-group", type=int, default=1)
    parser.add_argument("--frames-per-episode", type=int, default=6)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--predicate", action="append", default=[])
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--label-edges", action="store_true")
    parser.add_argument("--node-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--title", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--legend", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    out_dir = args.output_dir or (args.data_root / "inspection" / f"node_edge_overlay_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_dirs = args.episode_dir or select_examples(iter_episode_dirs(args.data_root), args.per_group)
    predicates = set(args.predicate) if args.predicate else None

    written = []
    summary = []
    for episode_dir in episode_dirs:
        meta = read_json(episode_dir / "metadata.json")
        data = np.load(episode_dir / "frames.npz")
        positions = load_position_records(episode_dir)
        rgbs = data["rgb"]
        chosen = frame_indices(int(rgbs.shape[0]), args.frames_per_episode)
        images = []
        for frame_idx in chosen:
            if frame_idx not in positions:
                raise KeyError(f"missing position record for {episode_dir} frame {frame_idx}")
            image = draw_frame(
                rgbs[frame_idx],
                meta,
                frame_idx,
                positions[frame_idx],
                predicates=predicates,
                max_edges=args.max_edges,
                label_edges=args.label_edges,
                node_labels=args.node_labels,
                draw_title=args.title,
                draw_legend=args.legend,
            )
            frame_out = out_dir / "frames" / f"{meta['episode_id']}_t{frame_idx:06d}.png"
            frame_out.parent.mkdir(parents=True, exist_ok=True)
            image.save(frame_out)
            written.append(str(frame_out))
            images.append(image)
        sheet = make_sheet(images, args.columns)
        sheet_out = out_dir / f"{meta['policy_id']}_success-{meta['episode_success']}_task-{meta['task_id']}_{meta['episode_id']}.png"
        sheet.save(sheet_out)
        written.append(str(sheet_out))
        summary.append(
            {
                "episode_dir": str(episode_dir),
                "sheet": str(sheet_out),
                "policy_id": meta.get("policy_id"),
                "task_id": meta.get("task_id"),
                "episode_success": meta.get("episode_success"),
                "frames": chosen,
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(written)} files to {out_dir}")
    for row in summary:
        print(row["sheet"])


if __name__ == "__main__":
    main()
