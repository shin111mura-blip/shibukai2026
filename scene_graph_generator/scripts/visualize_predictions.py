#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

try:
    from .common import DEFAULT_OUTPUT_ROOT
except ImportError:  # pragma: no cover - CLI path execution
    from common import DEFAULT_OUTPUT_ROOT
from scene_graph_generator.graph_generator.feature_cache import read_jsonl
from scene_graph_generator.graph_generator.schema import compact_graph, graph_triplets, read_json, write_json


def wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if sum(len(x) for x in current) + len(current) + len(word) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def draw_text_block(draw, x: int, y: int, title: str, lines: list[str], fill: tuple[int, int, int]) -> int:
    draw.text((x, y), title, fill=fill)
    y += 18
    for line in lines:
        draw.text((x, y), line, fill=(25, 25, 25))
        y += 14
    return y + 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--architecture", choices=["pooled_mlp", "node_query_decoder"], default="pooled_mlp")
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    metrics_path = args.output_root / "metrics" / args.architecture / f"{args.split}_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing evaluation metrics: {metrics_path}")
    pred_root = args.output_root / "predictions" / args.architecture / args.split
    if not pred_root.exists():
        raise FileNotFoundError(f"Missing predictions: {pred_root}")

    manifest_rows = [r for r in read_jsonl(args.output_root / "feature_cache" / "all_frames" / "cache_manifest.jsonl") if r["split"] == args.split]
    rng = random.Random(args.seed)
    selected = rng.sample(manifest_rows, min(args.count, len(manifest_rows)))
    out_dir = args.output_root / "visualizations" / args.architecture / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in selected:
        pred_path = pred_root / f"task_{row['task_id']:02d}" / f"global_{row['global_episode_index']:06d}" / f"{row['frame_index']:06d}.json"
        if not pred_path.exists():
            continue
        gt = compact_graph(read_json(Path(row["graph_path"])))
        pred = read_json(pred_path)
        gt_edges = set(graph_triplets(gt))
        pred_edges = set(graph_triplets(pred))
        tp = sorted(gt_edges & pred_edges)
        fp = sorted(pred_edges - gt_edges)
        fn = sorted(gt_edges - pred_edges)
        img = Image.open(row["image_path"]).convert("RGB").resize((384, 384))
        canvas = Image.new("RGB", (900, 420), (248, 248, 246))
        canvas.paste(img, (18, 18))
        draw = ImageDraw.Draw(canvas)
        y = 18
        title = f"task {row['task_id']:02d}  episode {row['global_episode_index']:06d}  frame {row['frame_index']:06d}"
        draw.text((424, y), title, fill=(0, 0, 0))
        y += 24
        for line in wrap(row["instruction"], 64):
            draw.text((424, y), line, fill=(40, 40, 40))
            y += 14
        y += 8
        y = draw_text_block(draw, 424, y, f"TP {len(tp)}", [f"{s} {p} {o}" for s, p, o in tp[:8]], (24, 125, 72))
        y = draw_text_block(draw, 424, y, f"FP {len(fp)}", [f"{s} {p} {o}" for s, p, o in fp[:8]], (183, 80, 24))
        y = draw_text_block(draw, 424, y, f"FN {len(fn)}", [f"{s} {p} {o}" for s, p, o in fn[:8]], (180, 32, 45))
        out_path = out_dir / f"task_{row['task_id']:02d}_global_{row['global_episode_index']:06d}_frame_{row['frame_index']:06d}.png"
        canvas.save(out_path)
        rows.append(
            {
                "image": str(out_path),
                "prediction": str(pred_path),
                "ground_truth": row["graph_path"],
                "task_id": row["task_id"],
                "global_episode_index": row["global_episode_index"],
                "frame_index": row["frame_index"],
                "tp": len(tp),
                "fp": len(fp),
                "fn": len(fn),
            }
        )

    report = {
        "status": "ok",
        "architecture": args.architecture,
        "split": args.split,
        "count_requested": args.count,
        "count_written": len(rows),
        "visualization_dir": str(out_dir),
        "examples": rows,
    }
    write_json(args.output_root / "visualizations" / args.architecture / f"{args.split}_visualization_status.json", report)
    print(json.dumps({"status": "ok", "written": len(rows), "dir": str(out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
