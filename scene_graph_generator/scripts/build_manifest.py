#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

try:
    from .common import DEFAULT_DEMO_MANIFEST, DEFAULT_GRAPH_ROOT, DEFAULT_OUTPUT_ROOT, load_demo_manifest
except ImportError:  # pragma: no cover - CLI path execution
    from common import DEFAULT_DEMO_MANIFEST, DEFAULT_GRAPH_ROOT, DEFAULT_OUTPUT_ROOT, load_demo_manifest
from scene_graph_generator.graph_generator.feature_cache import write_jsonl
from scene_graph_generator.graph_generator.schema import file_sha256, iter_graph_paths, parse_graph_path, write_json


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(graph_root: Path, demo_manifest: Path, split_path: Path, output_root: Path) -> dict:
    demos = load_demo_manifest(demo_manifest)
    split = __import__("json").load(open(split_path))
    ep_to_split = {}
    for split_name, eps in split["episodes"].items():
        for ep in eps:
            ep_to_split[int(ep)] = split_name
    rows = []
    missing = []
    seen = set()
    split_rows = {"train": [], "validation": [], "test": []}
    for path in iter_graph_paths(graph_root):
        task_id, episode_id, frame_idx = parse_graph_path(path)
        key = (task_id, episode_id, frame_idx)
        if key in seen:
            raise ValueError(f"Duplicate manifest key {key}")
        seen.add(key)
        if episode_id not in demos:
            missing.append(str(path))
            continue
        split_name = ep_to_split.get(episode_id)
        if split_name is None:
            missing.append(str(path))
            continue
        instruction = demos[episode_id]["language_instruction"]
        row = {
            "task_id": task_id,
            "global_episode_index": episode_id,
            "frame_index": frame_idx,
            "instruction": instruction,
            "graph_path": str(path),
            "split": split_name,
            "image_hash": None,
            "instruction_hash": sha_text(instruction),
            "graph_hash": file_sha256(path),
        }
        rows.append(row)
        split_rows[split_name].append(row)
    manifests = output_root / "manifests"
    write_jsonl(manifests / "all_frames.jsonl", rows)
    for split_name, split_data in split_rows.items():
        write_jsonl(manifests / f"{split_name}_frames.jsonl", split_data)
    summary = {
        "total_frames": len(rows),
        "split_frames": {k: len(v) for k, v in split_rows.items()},
        "duplicate_count": len(rows) - len(seen),
        "missing_count": len(missing),
        "missing": missing[:20],
        "image_hash_policy": "None on host build; fill during RLDS feature extraction because RGB is read only as OpenVLA input.",
    }
    write_json(output_root / "reports" / "manifest_audit.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    ap.add_argument("--demo-manifest", type=Path, default=DEFAULT_DEMO_MANIFEST)
    ap.add_argument("--split-path", type=Path, default=DEFAULT_OUTPUT_ROOT / "splits" / "split_seed42_train50_val25_test25.json")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = ap.parse_args()
    print(build(args.graph_root, args.demo_manifest, args.split_path, args.output_root))


if __name__ == "__main__":
    main()
