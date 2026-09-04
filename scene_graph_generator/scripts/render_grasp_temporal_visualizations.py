#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .common import DEFAULT_OUTPUT_ROOT
except ImportError:  # pragma: no cover
    from common import DEFAULT_OUTPUT_ROOT
from scene_graph_generator.graph_generator.feature_cache import read_jsonl
from scene_graph_generator.graph_generator.schema import compact_graph, graph_triplets, read_json, write_json


def triplet_lines(edges, limit=12):
    lines = [f"{s} {p} {o}" for s, p, o in sorted(edges)]
    if len(lines) > limit:
        return lines[:limit] + [f"... {len(lines) - limit} more"]
    return lines


def draw_block(draw, x, y, title, lines, color):
    draw.text((x, y), title, fill=color)
    y += 16
    for line in lines:
        draw.text((x, y), line[:60], fill=(25, 25, 25))
        y += 13
    return y + 8


def pred_path(root: Path, arch: str, split: str, row) -> Path:
    return root / "predictions" / arch / split / f"task_{row['task_id']:02d}" / f"global_{row['global_episode_index']:06d}" / f"{row['frame_index']:06d}.json"


def load_edges(path: Path):
    return set(graph_triplets(compact_graph(read_json(path))))


def render_sequence(root: Path, arch: str, rows, center_idx: int, out_path: Path, title: str):
    from PIL import Image, ImageDraw

    selected = rows[max(0, center_idx - 2) : min(len(rows), center_idx + 3)]
    w, h = 360, 520
    canvas = Image.new("RGB", (w * len(selected), h), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    for col, row in enumerate(selected):
        x0 = col * w
        img = Image.open(row["image_path"]).convert("RGB").resize((220, 220))
        canvas.paste(img, (x0 + 12, 42))
        pp = pred_path(root, arch, row["split"], row)
        gt_edges = load_edges(Path(row["graph_path"]))
        pred_edges = load_edges(pp) if pp.exists() else set()
        grasp_gt = [e for e in gt_edges if e[1] == "grasping"]
        grasp_pred = [e for e in pred_edges if e[1] == "grasping"]
        draw.text((x0 + 12, 12), f"t={row['frame_index']:04d} task={row['task_id']:02d}", fill=(0, 0, 0))
        y = 274
        y = draw_block(draw, x0 + 12, y, "GT grasp", triplet_lines(grasp_gt, 4) or ["none"], (25, 105, 65))
        y = draw_block(draw, x0 + 12, y, "Pred grasp", triplet_lines(grasp_pred, 4) or ["none"], (50, 70, 150))
        if col > 0:
            prev_gt = load_edges(Path(selected[col - 1]["graph_path"]))
            add = gt_edges - prev_gt
            rem = prev_gt - gt_edges
            y = draw_block(draw, x0 + 12, y, "GT add/remove", [f"+ {x}" for x in triplet_lines(add, 3)] + [f"- {x}" for x in triplet_lines(rem, 3)], (150, 80, 20))
    draw.text((12, h - 24), title[:180], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--architecture", default="pooled_mlp")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--max-grasp", type=int, default=6)
    ap.add_argument("--max-temporal", type=int, default=6)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.output_root / "feature_cache" / "all_frames" / "cache_manifest.jsonl") if r["split"] in set(args.splits.split(","))]
    by_ep = defaultdict(list)
    for row in rows:
        if pred_path(args.output_root, args.architecture, row["split"], row).exists():
            by_ep[(row["split"], row["task_id"], row["global_episode_index"])].append(row)
    for key in by_ep:
        by_ep[key].sort(key=lambda r: r["frame_index"])

    out_dir = args.output_root / "visualizations" / args.architecture / "grasp_temporal_2d"
    examples = []
    grasp_written = 0
    temporal_candidates = []
    for key, ep_rows in sorted(by_ep.items()):
        prev_edges = None
        for idx, row in enumerate(ep_rows):
            gt_edges = load_edges(Path(row["graph_path"]))
            if any(edge[1] == "grasping" for edge in gt_edges) and grasp_written < args.max_grasp:
                out_path = out_dir / f"grasp_{grasp_written:02d}_task_{row['task_id']:02d}_global_{row['global_episode_index']:06d}_frame_{row['frame_index']:06d}.png"
                render_sequence(args.output_root, args.architecture, ep_rows, idx, out_path, "Grasp relation sequence: GT and prediction around grasping frame")
                examples.append({"type": "grasp", "image": str(out_path), "row": row})
                grasp_written += 1
            if prev_edges is not None:
                change = len(gt_edges - prev_edges) + len(prev_edges - gt_edges)
                if change:
                    temporal_candidates.append((change, key, idx, row))
            prev_edges = gt_edges
    for n, (_change, key, idx, row) in enumerate(sorted(temporal_candidates, reverse=True)[: args.max_temporal]):
        ep_rows = by_ep[key]
        out_path = out_dir / f"temporal_{n:02d}_task_{row['task_id']:02d}_global_{row['global_episode_index']:06d}_frame_{row['frame_index']:06d}.png"
        render_sequence(args.output_root, args.architecture, ep_rows, idx, out_path, "Temporal relation-change sequence: GT added/removed triplets")
        examples.append({"type": "temporal", "image": str(out_path), "row": row})

    report = {"status": "ok", "architecture": args.architecture, "dir": str(out_dir), "examples": examples}
    write_json(out_dir / "grasp_temporal_manifest.json", report)
    print(json.dumps({"status": "ok", "dir": str(out_dir), "count": len(examples)}, sort_keys=True))


if __name__ == "__main__":
    main()
