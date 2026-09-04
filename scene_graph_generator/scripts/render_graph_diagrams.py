#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from .common import DEFAULT_OUTPUT_ROOT
except ImportError:  # pragma: no cover
    from common import DEFAULT_OUTPUT_ROOT
from scene_graph_generator.graph_generator.feature_cache import read_jsonl
from scene_graph_generator.graph_generator.schema import compact_graph, graph_node_ids, graph_triplets, read_json, write_json


PREDICATE_COLORS = {
    "left_of": (43, 96, 170),
    "right_of": (20, 125, 116),
    "above": (148, 73, 167),
    "below": (191, 88, 45),
    "front_of": (198, 150, 38),
    "behind": (99, 99, 99),
    "on": (38, 132, 66),
    "inside": (100, 87, 166),
    "contains": (157, 90, 65),
    "grasping": (190, 48, 60),
}


def short_node(node_id: str) -> str:
    replacements = {
        "akita_black_bowl": "black_bowl",
        "glazed_rim_porcelain_ramekin": "ramekin",
        "wooden_cabinet": "cabinet",
        "flat_stove": "stove",
    }
    out = node_id
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def arrow(draw, p1, p2, color, width=2):
    x1, y1 = p1
    x2, y2 = p2
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 8
    pts = [
        (x2, y2),
        (x2 - size * math.cos(angle - 0.45), y2 - size * math.sin(angle - 0.45)),
        (x2 - size * math.cos(angle + 0.45), y2 - size * math.sin(angle + 0.45)),
    ]
    draw.polygon(pts, fill=color)


def layout(nodes, x0, y0, w, h):
    cx = x0 + w / 2
    cy = y0 + h / 2
    rx = w * 0.36
    ry = h * 0.34
    positions = {}
    for idx, node_id in enumerate(nodes):
        angle = -math.pi / 2 + 2 * math.pi * idx / max(len(nodes), 1)
        positions[node_id] = (int(cx + rx * math.cos(angle)), int(cy + ry * math.sin(angle)))
    return positions


def draw_graph(draw, graph, x0, y0, w, h, title):
    nodes = list(graph_node_ids(graph))
    edges = list(graph_triplets(graph))
    draw.text((x0, y0), title, fill=(0, 0, 0))
    box_top = y0 + 24
    draw.rectangle((x0, box_top, x0 + w, y0 + h), outline=(210, 210, 210), width=1)
    positions = layout(nodes, x0 + 14, box_top + 6, w - 28, h - 150)
    for s, pred, o in edges:
        if s not in positions or o not in positions:
            continue
        color = PREDICATE_COLORS.get(pred, (80, 80, 80))
        x1, y1 = positions[s]
        x2, y2 = positions[o]
        dx = x2 - x1
        dy = y2 - y1
        dist = max((dx * dx + dy * dy) ** 0.5, 1)
        start = (int(x1 + 24 * dx / dist), int(y1 + 24 * dy / dist))
        end = (int(x2 - 24 * dx / dist), int(y2 - 24 * dy / dist))
        arrow(draw, start, end, color, width=2)
    for node_id, (x, y) in positions.items():
        draw.ellipse((x - 27, y - 18, x + 27, y + 18), fill=(255, 255, 255), outline=(35, 35, 35), width=2)
        label = short_node(node_id)
        draw.text((x - min(24, len(label) * 3), y - 6), label[:12], fill=(0, 0, 0))
    list_y = y0 + h - 112
    draw.text((x0 + 8, list_y), f"edges: {len(edges)}", fill=(0, 0, 0))
    list_y += 16
    for edge in edges[:8]:
        s, pred, o = edge
        color = PREDICATE_COLORS.get(pred, (80, 80, 80))
        draw.rectangle((x0 + 8, list_y + 3, x0 + 18, list_y + 13), fill=color)
        draw.text((x0 + 24, list_y), f"{short_node(s)} -> {pred} -> {short_node(o)}"[:58], fill=(25, 25, 25))
        list_y += 14
    if len(edges) > 8:
        draw.text((x0 + 24, list_y), f"... {len(edges) - 8} more", fill=(80, 80, 80))


def find_examples(output_root: Path, architecture: str, split: str, count: int):
    rows = [r for r in read_jsonl(output_root / "feature_cache" / "all_frames" / "cache_manifest.jsonl") if r["split"] == split]
    scored = []
    pred_root = output_root / "predictions" / architecture / split
    for row in rows:
        pred_path = pred_root / f"task_{row['task_id']:02d}" / f"global_{row['global_episode_index']:06d}" / f"{row['frame_index']:06d}.json"
        if not pred_path.exists():
            continue
        gt = compact_graph(read_json(Path(row["graph_path"])))
        pred = read_json(pred_path)
        gt_edges = set(graph_triplets(gt))
        pred_edges = set(graph_triplets(pred))
        fp = len(pred_edges - gt_edges)
        fn = len(gt_edges - pred_edges)
        scored.append((fp + fn, len(gt_edges) + len(pred_edges), row, gt, pred, pred_path))
    scored.sort(key=lambda x: (x[0], x[1], x[2]["task_id"], x[2]["global_episode_index"], x[2]["frame_index"]))
    return scored[:count]


def render_example(output_root: Path, architecture: str, split: str, example, out_dir: Path):
    from PIL import Image, ImageDraw

    _, _, row, gt, pred, pred_path = example
    img = Image.open(row["image_path"]).convert("RGB").resize((320, 320))
    canvas = Image.new("RGB", (1280, 760), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(img, (24, 72))
    title = (
        f"{architecture} / {split} / task {row['task_id']:02d} / "
        f"global {row['global_episode_index']:06d} / frame {row['frame_index']:06d}"
    )
    draw.text((24, 20), title, fill=(0, 0, 0))
    draw.text((24, 48), row["instruction"][:120], fill=(35, 35, 35))
    draw.text((24, 410), "RGB input used only by frozen OpenVLA feature extractor", fill=(60, 60, 60))
    gt_edges = set(graph_triplets(gt))
    pred_edges = set(graph_triplets(pred))
    draw.text(
        (24, 438),
        f"Graph comparison: TP={len(gt_edges & pred_edges)}  FP={len(pred_edges - gt_edges)}  FN={len(gt_edges - pred_edges)}",
        fill=(0, 0, 0),
    )
    draw_graph(draw, gt, 380, 72, 410, 640, "Teacher Scene Graph")
    draw_graph(draw, pred, 830, 72, 410, 640, "Predicted Scene Graph")
    out_path = out_dir / f"{architecture}_task_{row['task_id']:02d}_global_{row['global_episode_index']:06d}_frame_{row['frame_index']:06d}.png"
    canvas.save(out_path)
    return {
        "image": str(out_path),
        "prediction": str(pred_path),
        "ground_truth": row["graph_path"],
        "rgb": row["image_path"],
        "task_id": row["task_id"],
        "global_episode_index": row["global_episode_index"],
        "frame_index": row["frame_index"],
        "tp": len(gt_edges & pred_edges),
        "fp": len(pred_edges - gt_edges),
        "fn": len(gt_edges - pred_edges),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--architecture", choices=["pooled_mlp", "node_query_decoder"], default="pooled_mlp")
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--count", type=int, default=3)
    args = ap.parse_args()

    out_dir = args.output_root / "visualizations" / args.architecture / f"{args.split}_graph_diagrams"
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = find_examples(args.output_root, args.architecture, args.split, args.count)
    rows = [render_example(args.output_root, args.architecture, args.split, example, out_dir) for example in examples]
    report = {
        "status": "ok",
        "architecture": args.architecture,
        "split": args.split,
        "count": len(rows),
        "dir": str(out_dir),
        "examples": rows,
    }
    write_json(out_dir / "graph_diagram_manifest.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
