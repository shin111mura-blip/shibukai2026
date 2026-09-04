#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .common import DEFAULT_DEMO_MANIFEST, DEFAULT_GRAPH_ROOT, DEFAULT_OUTPUT_ROOT, load_demo_manifest
except ImportError:  # pragma: no cover - CLI path execution
    from common import DEFAULT_DEMO_MANIFEST, DEFAULT_GRAPH_ROOT, DEFAULT_OUTPUT_ROOT, load_demo_manifest
from scene_graph_generator.graph_generator.schema import iter_graph_paths, parse_graph_path, write_json


def split_counts(n: int) -> tuple[int, int, int]:
    if n >= 3:
        train = max(1, round(n * 0.50))
        val = max(1, round(n * 0.25))
        test = n - train - val
        if test < 1:
            test = 1
            train = max(1, n - val - test)
        while train + val + test > n:
            train -= 1
        while train + val + test < n:
            train += 1
        return train, val, test
    if n == 2:
        return 1, 1, 0
    return n, 0, 0


def create(graph_root: Path, demo_manifest: Path, output_root: Path, seed: int = 42) -> dict:
    demos = load_demo_manifest(demo_manifest)
    task_to_episodes = defaultdict(set)
    episode_frame_counts = Counter()
    for path in iter_graph_paths(graph_root):
        task_id, episode_id, _ = parse_graph_path(path)
        task_to_episodes[task_id].add(episode_id)
        episode_frame_counts[episode_id] += 1
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    per_task = {}
    for task_id in sorted(task_to_episodes):
        eps = sorted(task_to_episodes[task_id])
        rng.shuffle(eps)
        n_train, n_val, n_test = split_counts(len(eps))
        train_eps = sorted(eps[:n_train])
        val_eps = sorted(eps[n_train : n_train + n_val])
        test_eps = sorted(eps[n_train + n_val : n_train + n_val + n_test])
        splits["train"].extend(train_eps)
        splits["validation"].extend(val_eps)
        splits["test"].extend(test_eps)
        per_task[str(task_id)] = {
            "task_id": task_id,
            "task_name": demos[eps[0]].get("task_name") if eps and eps[0] in demos else None,
            "train": train_eps,
            "validation": val_eps,
            "test": test_eps,
            "episode_counts": {"train": len(train_eps), "validation": len(val_eps), "test": len(test_eps)},
            "frame_counts": {
                "train": sum(episode_frame_counts[e] for e in train_eps),
                "validation": sum(episode_frame_counts[e] for e in val_eps),
                "test": sum(episode_frame_counts[e] for e in test_eps),
            },
        }
    for key in splits:
        splits[key] = sorted(splits[key])
    split_obj = {
        "split": {
            "unit": "global_episode_index",
            "stratify_by": "task_id",
            "train_ratio": 0.50,
            "validation_ratio": 0.25,
            "test_ratio": 0.25,
            "seed": seed,
        },
        "episodes": splits,
        "tasks": per_task,
    }
    split_hash = hashlib.sha256(json.dumps(split_obj, sort_keys=True).encode()).hexdigest()
    split_obj["split_hash"] = split_hash
    overlap = {
        "train_validation": sorted(set(splits["train"]) & set(splits["validation"])),
        "train_test": sorted(set(splits["train"]) & set(splits["test"])),
        "validation_test": sorted(set(splits["validation"]) & set(splits["test"])),
    }
    assigned = set(splits["train"]) | set(splits["validation"]) | set(splits["test"])
    all_eps = set(episode_frame_counts)
    summary = {
        "seed": seed,
        "split_hash": split_hash,
        "episode_counts": {k: len(v) for k, v in splits.items()},
        "frame_counts": {k: sum(episode_frame_counts[e] for e in v) for k, v in splits.items()},
        "task_episode_counts": {task: val["episode_counts"] for task, val in per_task.items()},
        "task_frame_counts": {task: val["frame_counts"] for task, val in per_task.items()},
        "overlap_check": overlap,
        "has_overlap": any(overlap.values()),
        "unassigned_episode_count": len(all_eps - assigned),
        "unassigned_episodes": sorted(all_eps - assigned),
        "total_episode_count": len(all_eps),
        "total_frame_count": sum(episode_frame_counts.values()),
    }
    out = output_root / "splits"
    write_json(out / "split_seed42_train50_val25_test25.json", split_obj)
    write_json(out / "split_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    ap.add_argument("--demo-manifest", type=Path, default=DEFAULT_DEMO_MANIFEST)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    summary = create(args.graph_root, args.demo_manifest, args.output_root, args.seed)
    print(json.dumps(summary["episode_counts"], sort_keys=True), json.dumps(summary["frame_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
