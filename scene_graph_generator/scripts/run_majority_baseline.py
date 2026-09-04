#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .common import DEFAULT_OUTPUT_ROOT
except ImportError:  # pragma: no cover - CLI path execution
    from common import DEFAULT_OUTPUT_ROOT
from scene_graph_generator.graph_generator.feature_cache import read_jsonl
from scene_graph_generator.graph_generator.metrics import summarize_examples
from scene_graph_generator.graph_generator.schema import compact_graph, graph_node_ids, graph_triplets, read_json, write_json


def run(output_root: Path) -> dict:
    train = read_jsonl(output_root / "manifests" / "train_frames.jsonl")
    all_rows = {split: read_jsonl(output_root / "manifests" / f"{split}_frames.jsonl") for split in ("validation", "test")}
    counters = defaultdict(Counter)
    graphs_by_key = {}
    for row in train:
        graph = compact_graph(read_json(Path(row["graph_path"])))
        key = json.dumps({"nodes": graph_node_ids(graph), "edges": graph_triplets(graph)}, sort_keys=True)
        counters[int(row["task_id"])][key] += 1
        graphs_by_key[key] = graph
    majority = {}
    for task_id, counter in counters.items():
        key, count = counter.most_common(1)[0]
        majority[task_id] = graphs_by_key[key]
        majority[task_id]["majority_train_count"] = count
    pred_root = output_root / "predictions" / "majority_graph"
    metric_root = output_root / "metrics" / "majority_graph"
    write_json(metric_root / "majority_graphs_by_task.json", {str(k): v for k, v in majority.items()})
    metrics = {}
    for split, rows in all_rows.items():
        examples = []
        for row in rows:
            pred = compact_graph(majority[int(row["task_id"])])
            pred.pop("majority_train_count", None)
            out_path = pred_root / f"task_{row['task_id']:02d}" / f"global_{row['global_episode_index']:06d}" / f"{row['frame_index']:06d}.json"
            write_json(out_path, pred)
            gt = compact_graph(read_json(Path(row["graph_path"])))
            examples.append(
                {
                    "pred_nodes": graph_node_ids(pred),
                    "gt_nodes": graph_node_ids(gt),
                    "pred_edges": graph_triplets(pred),
                    "gt_edges": graph_triplets(gt),
                }
            )
        metrics[split] = summarize_examples(examples)
        write_json(metric_root / f"{split}_metrics.json", metrics[split])
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = ap.parse_args()
    metrics = run(args.output_root)
    print(json.dumps({k: v["triplet"]["f1"] for k, v in metrics.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
