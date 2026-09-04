#!/usr/bin/env python3
"""Visualize generated oracle graphs as RGB overlays or graph-only diagrams."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from oracle_scene_graph_utils import read_jsonl


RELATION_COLORS = {
    "left_of": (120, 120, 120),
    "right_of": (120, 120, 120),
    "front_of": (120, 120, 120),
    "behind": (120, 120, 120),
    "above": (120, 120, 120),
    "below": (120, 120, 120),
    "near": (80, 160, 255),
    "next_to": (80, 160, 255),
    "between": (120, 120, 120),
    "on": (50, 190, 90),
    "inside": (200, 120, 40),
    "touching": (255, 70, 70),
    "grasped_by": (170, 70, 255),
    "grasping": (170, 70, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-dir", type=Path, default=Path("outputs/scene_graph_probe/graphs"))
    parser.add_argument("--rgb-dir", type=Path, default=Path("outputs/scene_graph_probe/rgb"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_probe/overlays"))
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument("--no-flip-vertical", action="store_true", help="Disable the default vertical RGB image correction.")
    parser.add_argument("--no-flip-horizontal", action="store_true", help="Disable the default horizontal RGB image correction.")
    parser.add_argument(
        "--flip-projection-vertical",
        action="store_true",
        help="Also flip projected graph y coordinates. Usually leave this off for LIBERO agentview.",
    )
    parser.add_argument(
        "--flip-projection-horizontal",
        action="store_true",
        help="Also flip projected graph x coordinates. Enabled by default for LIBERO rollout-video orientation.",
    )
    parser.add_argument(
        "--no-flip-projection-horizontal",
        dest="flip_projection_horizontal",
        action="store_false",
        help="Disable the default horizontal projection correction.",
    )
    parser.set_defaults(flip_projection_horizontal=True)
    return parser.parse_args()


def iter_graph_paths(graphs_dir: Path) -> Iterable[Path]:
    yield from sorted({*graphs_dir.glob("episode_*.jsonl"), *graphs_dir.glob("trial_*.jsonl"), *graphs_dir.glob("*.jsonl")})


def resolve_rgb_path(explicit: Optional[str], fallback: Path) -> Path:
    if not explicit:
        return fallback
    path = Path(explicit)
    if path.exists():
        return path
    marker = "/workspace/"
    explicit_str = str(explicit)
    if explicit_str.startswith(marker):
        host_path = Path(explicit_str.replace(marker, "", 1))
        if host_path.exists():
            return host_path
    return path


def node_positions(
    record: Dict,
    width: int,
    height: int,
    flip_projection_vertical: bool = False,
    flip_projection_horizontal: bool = False,
) -> Dict[str, Tuple[int, int]]:
    nodes = record.get("nodes", [])
    projected = {
        n.get("id"): tuple(n.get("center2d"))
        for n in nodes
        if n.get("center2d") is not None and len(n.get("center2d")) == 2
    }
    if projected:
        image_w = record.get("metadata", {}).get("camera_projection", {}).get("image_width") or 128
        image_h = record.get("metadata", {}).get("camera_projection", {}).get("image_height") or 128
        positions = {}
        for node_id, point in projected.items():
            x = int(point[0] * width / image_w)
            y = int(point[1] * height / image_h)
            positions[node_id] = (
                width - 1 - x if flip_projection_horizontal else x,
                height - 1 - y if flip_projection_vertical else y,
            )
        return positions
    object_nodes = [n for n in nodes if n.get("type") == "object"]
    positions: Dict[str, Tuple[int, int]] = {}
    if not object_nodes:
        return positions
    xy = np.asarray([n["pos_world"][:2] for n in object_nodes if n.get("pos_world") is not None], dtype=float)
    if xy.size == 0:
        return positions
    low = xy.min(axis=0)
    high = xy.max(axis=0)
    span = np.maximum(high - low, 1e-4)
    for node in object_nodes:
        if node.get("pos_world") is None:
            continue
        point = (np.asarray(node["pos_world"][:2], dtype=float) - low) / span
        positions[node["id"]] = (int(80 + point[0] * 420), int(420 - point[1] * 320))
    gripper = next((n for n in nodes if n.get("id") == "gripper" and n.get("pos_world") is not None), None)
    if gripper is not None:
        point = (np.asarray(gripper["pos_world"][:2], dtype=float) - low) / span
        positions["gripper"] = (int(80 + point[0] * 420), int(420 - point[1] * 320))
    return positions


def draw_record(
    record: Dict,
    rgb_path: Optional[Path],
    out_path: Path,
    flip_vertical: bool = True,
    flip_horizontal: bool = True,
    flip_projection_vertical: bool = False,
    flip_projection_horizontal: bool = False,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False

    if rgb_path and rgb_path.exists():
        image = Image.open(rgb_path).convert("RGB")
        if flip_vertical:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if flip_horizontal:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        canvas = image.resize((512, 512))
    else:
        canvas = Image.new("RGB", (640, 480), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    positions = node_positions(
        record,
        canvas.width,
        canvas.height,
        flip_projection_vertical=flip_projection_vertical,
        flip_projection_horizontal=flip_projection_horizontal,
    )

    for edge in record.get("edges", []):
        src = edge.get("src")
        dst = edge.get("dst")
        dsts = dst if isinstance(dst, list) else [dst]
        color = RELATION_COLORS.get(edge.get("rel"), (90, 90, 90))
        for dst_id in dsts:
            if src in positions and dst_id in positions:
                width = 3 if edge.get("rel") in {"touching", "grasping", "grasped_by"} else 1
                draw.line([positions[src], positions[dst_id]], fill=color, width=width)

    for node in record.get("nodes", []):
        node_id = node.get("id")
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        fill = (40, 40, 40) if node.get("type") == "object" else (160, 40, 180)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=fill)
        label = node_id[:30]
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_x = min(max(x + 9, 1), max(1, canvas.width - text_w - 3))
        label_y = min(max(y - 7, 21), max(21, canvas.height - text_h - 3))
        draw.rectangle((label_x - 2, label_y - 2, label_x + text_w + 2, label_y + text_h + 2), fill=(255, 255, 255))
        draw.text((label_x, label_y), label, fill=(20, 20, 20), font=font)

    title = f"task={record.get('task_id')} episode={record.get('episode_id')} t={record.get('timestep')} edges={len(record.get('edges', []))}"
    draw.rectangle((0, 0, canvas.width, 20), fill=(255, 255, 255))
    draw.text((6, 5), title, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return True


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for graph_path in iter_graph_paths(args.graphs_dir):
        episode = graph_path.stem
        for record in read_jsonl(graph_path):
            if count >= args.max_images:
                break
            timestep = int(record.get("timestep", 0))
            rgb_path = None
            if not args.graph_only:
                explicit = record.get("rgb_path")
                fallback = args.rgb_dir / episode / f"t{timestep:06d}.png"
                rgb_path = resolve_rgb_path(explicit, fallback)
            out_path = args.output_dir / episode / f"t{timestep:06d}_overlay.png"
            if draw_record(
                record,
                rgb_path,
                out_path,
                flip_vertical=not args.no_flip_vertical,
                flip_horizontal=not args.no_flip_horizontal,
                flip_projection_vertical=args.flip_projection_vertical,
                flip_projection_horizontal=args.flip_projection_horizontal,
            ):
                count += 1
        if count >= args.max_images:
            break
    summary = args.output_dir / "overlay_summary.txt"
    summary.write_text(
        f"generated_overlays={count}\n"
        f"flip_rgb_vertical={not args.no_flip_vertical}\n"
        f"flip_rgb_horizontal={not args.no_flip_horizontal}\n"
        f"flip_projection_vertical={args.flip_projection_vertical}\n"
        f"flip_projection_horizontal={args.flip_projection_horizontal}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
