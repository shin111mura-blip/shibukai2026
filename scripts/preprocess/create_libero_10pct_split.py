#!/usr/bin/env python3
"""Create a deterministic 10% demonstration split for LIBERO-Spatial RLDS data."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def select_demo_ids(demo_ids: List[int], fraction: float, seed: int) -> List[int]:
    rng = random.Random(seed)
    ordered = list(demo_ids)
    count = max(1, int(math.ceil(len(ordered) * fraction)))
    selected = rng.sample(ordered, count)
    return sorted(selected)


def _decode_first(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if hasattr(value, "shape"):
        if value.shape == ():
            return _decode_first(value.item())
        if len(value) > 0:
            return _decode_first(value[0])
    return ""


def _load_tasks_from_tfds(data_root: Path, dataset_name: str) -> Dict[str, Dict[str, Any]]:
    import dlimp as dl
    import tensorflow_datasets as tfds

    builder = tfds.builder(dataset_name, data_dir=str(data_root))
    ds = dl.DLataset.from_rlds(builder, split="train", shuffle=False)
    tasks: Dict[str, Dict[str, Any]] = {}
    for global_idx, episode in ds.enumerate().as_numpy_iterator():
        global_idx = int(global_idx)
        language = _decode_first(episode.get("language_instruction", b""))
        if not language and "steps" in episode:
            steps = episode["steps"]
            first = next(iter(steps))
            language = _decode_first(first.get("language_instruction", b""))
        task_key = language or "unknown_task"
        if task_key not in tasks:
            tasks[task_key] = {
                "task_id": len(tasks),
                "task_name": task_key,
                "global_episode_indices": [],
            }
        tasks[task_key]["global_episode_indices"].append(int(global_idx))
    return tasks


def create_split(
    data_root: Path,
    dataset_name: str,
    suite: str,
    fraction: float,
    seed: int,
) -> Dict[str, Any]:
    tasks = _load_tasks_from_tfds(data_root, dataset_name)
    payload_tasks = {}
    for task_name, task in sorted(tasks.items(), key=lambda kv: kv[1]["task_id"]):
        demo_ids = task["global_episode_indices"]
        selected = select_demo_ids(demo_ids, fraction=fraction, seed=seed + int(task["task_id"]))
        payload_tasks[str(task["task_id"])] = {
            "task_id": int(task["task_id"]),
            "task_name": task_name,
            "total_demonstrations": len(demo_ids),
            "global_episode_indices": demo_ids,
            "selected_demonstration_ids": selected,
            "selected_global_episode_indices": selected,
        }
    return {
        "suite": suite,
        "dataset_name": dataset_name,
        "seed": seed,
        "fraction": fraction,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Group trajectories by task language, sample ceil(fraction * demos) with minimum 1 per task.",
        "tasks": payload_tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/modified_libero_rlds"))
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("splits/libero_spatial_10pct_seed42.json"))
    args = parser.parse_args()

    split = create_split(args.data_root, args.dataset_name, args.suite, args.fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
