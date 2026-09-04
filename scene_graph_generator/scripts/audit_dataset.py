#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .common import DEFAULT_DEMO_MANIFEST, DEFAULT_GRAPH_ROOT, DEFAULT_OUTPUT_ROOT, load_demo_manifest
except ImportError:  # pragma: no cover - CLI path execution
    from common import DEFAULT_DEMO_MANIFEST, DEFAULT_GRAPH_ROOT, DEFAULT_OUTPUT_ROOT, load_demo_manifest
from scene_graph_generator.graph_generator.schema import FORBIDDEN_PREDICATES, iter_graph_paths, parse_graph_path, read_json, validate_graph, write_json


def audit(graph_root: Path, demo_manifest: Path) -> dict:
    demos = load_demo_manifest(demo_manifest)
    episode_frames = defaultdict(int)
    task_episode = {}
    task_frames = Counter()
    predicates = Counter()
    node_ids = Counter()
    errors = []
    seen = set()
    for path in iter_graph_paths(graph_root):
        task_id, episode_id, frame_idx = parse_graph_path(path)
        key = (task_id, episode_id, frame_idx)
        if key in seen:
            errors.append(f"duplicate frame key {key}")
        seen.add(key)
        graph = read_json(path)
        graph_errors = validate_graph(graph)
        if graph_errors:
            errors.extend(f"{path}: {e}" for e in graph_errors[:10])
        for edge in graph.get("binary_edges", []):
            predicates[edge["predicate"]] += 1
            if edge["predicate"] in FORBIDDEN_PREDICATES:
                errors.append(f"{path}: forbidden predicate {edge['predicate']}")
        for node in graph.get("nodes", []):
            node_ids[node["id"]] += 1
        episode_frames[episode_id] += 1
        task_episode[episode_id] = task_id
        task_frames[task_id] += 1
    missing_demo = sorted(set(episode_frames) - set(demos))
    missing_graph_episode = sorted(set(demos) - set(episode_frames))
    if missing_demo:
        errors.append(f"episodes in graphs but not demo manifest: {missing_demo[:10]}")
    if missing_graph_episode:
        errors.append(f"episodes in demo manifest but not graphs: {missing_graph_episode[:10]}")
    task_episode_counts = Counter(task_episode.values())
    return {
        "graph_root": str(graph_root),
        "demo_manifest": str(demo_manifest),
        "episode_count": len(episode_frames),
        "frame_count": len(seen),
        "task_count": len(task_episode_counts),
        "task_episode_counts": {str(k): int(v) for k, v in sorted(task_episode_counts.items())},
        "task_frame_counts": {str(k): int(v) for k, v in sorted(task_frames.items())},
        "predicate_counts": dict(sorted(predicates.items())),
        "node_id_count": len(node_ids),
        "missing_graph_count": 0,
        "duplicate_count": len(seen) - len(set(seen)),
        "errors": errors,
        "passed": not errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    ap.add_argument("--demo-manifest", type=Path, default=DEFAULT_DEMO_MANIFEST)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = ap.parse_args()
    result = audit(args.graph_root, args.demo_manifest)
    write_json(args.output_root / "reports" / "dataset_audit.json", result)
    print(f"episodes={result['episode_count']} frames={result['frame_count']} passed={result['passed']}")


if __name__ == "__main__":
    main()
