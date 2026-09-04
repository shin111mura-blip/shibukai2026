#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .common import DEFAULT_OUTPUT_ROOT
except ImportError:  # pragma: no cover - CLI path execution
    from common import DEFAULT_OUTPUT_ROOT
from scene_graph_generator.graph_generator.feature_cache import read_jsonl
from scene_graph_generator.graph_generator.metrics import prf
from scene_graph_generator.graph_generator.schema import compact_graph, graph_triplets, read_json, write_json


def change_set(prev_edges: set[tuple[str, str, str]], curr_edges: set[tuple[str, str, str]]) -> set[tuple[str, str, str, str]]:
    added = {("add", *edge) for edge in curr_edges - prev_edges}
    removed = {("remove", *edge) for edge in prev_edges - curr_edges}
    return added | removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--architecture", choices=["pooled_mlp", "node_query_decoder"], default="pooled_mlp")
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    args = ap.parse_args()

    pred_root = args.output_root / "predictions" / args.architecture / args.split
    if not pred_root.exists():
        raise FileNotFoundError(f"Missing predictions: {pred_root}")
    rows = [r for r in read_jsonl(args.output_root / "feature_cache" / "all_frames" / "cache_manifest.jsonl") if r["split"] == args.split]
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[(row["task_id"], row["global_episode_index"])].append(row)
    totals = defaultdict(int)
    per_task = defaultdict(lambda: defaultdict(int))
    missing_predictions = 0
    transition_examples = []
    for (task_id, episode), episode_rows in sorted(by_episode.items()):
        prev_gt = None
        prev_pred = None
        for row in sorted(episode_rows, key=lambda x: x["frame_index"]):
            pred_path = pred_root / f"task_{row['task_id']:02d}" / f"global_{row['global_episode_index']:06d}" / f"{row['frame_index']:06d}.json"
            if not pred_path.exists():
                missing_predictions += 1
                prev_gt = None
                prev_pred = None
                continue
            gt_edges = set(graph_triplets(compact_graph(read_json(Path(row["graph_path"])))))
            pred_edges = set(graph_triplets(read_json(pred_path)))
            if prev_gt is not None and prev_pred is not None:
                gt_change = change_set(prev_gt, gt_edges)
                pred_change = change_set(prev_pred, pred_edges)
                tp = len(gt_change & pred_change)
                fp = len(pred_change - gt_change)
                fn = len(gt_change - pred_change)
                for scope in (totals, per_task[task_id]):
                    scope["tp"] += tp
                    scope["fp"] += fp
                    scope["fn"] += fn
                    scope["transitions"] += 1
                    scope["gt_changed_transitions"] += int(bool(gt_change))
                    scope["pred_changed_transitions"] += int(bool(pred_change))
                    scope["exact_change_match"] += int(gt_change == pred_change)
                if gt_change or pred_change:
                    transition_examples.append(
                        {
                            "task_id": task_id,
                            "global_episode_index": episode,
                            "frame_index": row["frame_index"],
                            "gt_changes": sorted(gt_change)[:20],
                            "pred_changes": sorted(pred_change)[:20],
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                        }
                    )
            prev_gt = gt_edges
            prev_pred = pred_edges

    metrics = prf(totals["tp"], totals["fp"], totals["fn"])
    report = {
        "status": "ok",
        "architecture": args.architecture,
        "split": args.split,
        "missing_predictions": missing_predictions,
        "num_episodes": len(by_episode),
        "num_transitions": totals["transitions"],
        "gt_changed_transitions": totals["gt_changed_transitions"],
        "pred_changed_transitions": totals["pred_changed_transitions"],
        "exact_change_match": totals["exact_change_match"] / totals["transitions"] if totals["transitions"] else 0.0,
        "triplet_change": metrics,
        "per_task": {
            f"task_{task_id:02d}": {
                "num_transitions": row["transitions"],
                "gt_changed_transitions": row["gt_changed_transitions"],
                "pred_changed_transitions": row["pred_changed_transitions"],
                "exact_change_match": row["exact_change_match"] / row["transitions"] if row["transitions"] else 0.0,
                "triplet_change": prf(row["tp"], row["fp"], row["fn"]),
            }
            for task_id, row in sorted(per_task.items())
        },
        "examples": transition_examples[:50],
    }
    out_path = args.output_root / "metrics" / args.architecture / f"{args.split}_dynamic_changes.json"
    write_json(out_path, report)
    md = [
        f"# {args.architecture} {args.split} Dynamic Change Analysis",
        "",
        f"- transitions: {report['num_transitions']}",
        f"- gt_changed_transitions: {report['gt_changed_transitions']}",
        f"- pred_changed_transitions: {report['pred_changed_transitions']}",
        f"- exact_change_match: {report['exact_change_match']:.6f}",
        f"- triplet_change_f1: {metrics['f1']:.6f}",
        f"- precision: {metrics['precision']:.6f}",
        f"- recall: {metrics['recall']:.6f}",
        f"- missing_predictions: {missing_predictions}",
        "",
    ]
    (args.output_root / "reports" / f"{args.architecture}_{args.split}_dynamic_changes.md").write_text("\n".join(md))
    print(json.dumps({"status": "ok", "output": str(out_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
