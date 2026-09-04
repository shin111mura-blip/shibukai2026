#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap

bootstrap()

from scene_graph.canonicalize import sha256_payload
from scene_graph.node_extractor import save_rgb
from scene_graph.rule_generator import (
    build_frame_graphs,
    create_env,
    default_rule_config,
    h5_attr_text,
    import_libero,
    resolve_task,
    set_demo_state,
    sorted_demo_keys,
    write_json,
)


DEFAULT_COUNTS_CSV = Path(
    "outputs/openvla_libero_spatial_lora_pct_sweep_subsetseed42_rerun/dataset_audit/full_counts.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/scene_graph_gt_openvla_spatial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rule-based SceneGraph GT for all OpenVLA LIBERO-Spatial no-noops demos."
    )
    parser.add_argument("--full-counts-csv", type=Path, default=DEFAULT_COUNTS_CSV)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mapping-jsonl",
        type=Path,
        default=None,
        help="Mapping from OpenVLA RLDS global episodes to local HDF5 demos. Defaults to OUTPUT_DIR/reports/openvla_hdf5_mapping.jsonl.",
    )
    parser.add_argument(
        "--allow-rank-demo-fallback",
        action="store_true",
        help="Debug only: infer demo index from task-local global episode rank when no mapping exists.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--limit-demos", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-observable", action="store_true", default=True)
    parser.add_argument("--no-save-observable", dest="save_observable", action="store_false")
    parser.add_argument("--save-diagnostics", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    return parser.parse_args()


def read_counts_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "global_episode_index": int(row["global_episode_index"]),
                    "task_id": int(row["task_id"]),
                    "task_name": row["task_name"],
                    "language_instruction": row["language_instruction"],
                    "num_steps": int(row["num_steps"]),
                    "episode_length": int(row.get("episode_length") or row["num_steps"]),
                    "episode_id": row.get("episode_id", ""),
                }
            )
    return sorted(rows, key=lambda row: int(row["global_episode_index"]))


def build_task_lookup(suite: str, rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    _benchmark, _get_libero_path, _classes = import_libero()
    lookup: dict[int, dict[str, Any]] = {}
    for task_id in sorted({int(row["task_id"]) for row in rows}):
        language = next(row["language_instruction"] for row in rows if int(row["task_id"]) == task_id)
        task = resolve_task(suite, language)
        task["openvla_task_id"] = task_id
        task["libero_benchmark_task_id"] = int(task["task_id"])
        lookup[task_id] = task
    return lookup


def attach_demo_indices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[int(row["task_id"])].append(row)
    for task_rows in by_task.values():
        for demo_index, row in enumerate(sorted(task_rows, key=lambda item: int(item["global_episode_index"]))):
            row["libero_demo_index"] = demo_index
            row["libero_demo_key"] = f"demo_{demo_index}"
            row["demo_id"] = f"global_{int(row['global_episode_index']):06d}"
    return sorted(rows, key=lambda row: int(row["global_episode_index"]))


def read_mapping_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[int(row["global_episode_index"])] = row
    return mapping


def attach_demo_indices_from_mapping(rows: list[dict[str, Any]], mapping_path: Path) -> list[dict[str, Any]]:
    mapping = read_mapping_jsonl(mapping_path)
    attached: list[dict[str, Any]] = []
    for row0 in rows:
        row = dict(row0)
        global_index = int(row["global_episode_index"])
        if global_index not in mapping:
            raise KeyError(f"global episode {global_index} is missing from mapping {mapping_path}")
        mapped = mapping[global_index]
        row["libero_demo_index"] = int(mapped["libero_demo_index"])
        row["libero_demo_key"] = str(mapped["libero_demo_key"])
        row["demo_id"] = str(mapped.get("demo_id") or f"global_{global_index:06d}")
        row["libero_benchmark_task_id"] = int(mapped["libero_benchmark_task_id"])
        row["hdf5_num_steps"] = int(mapped["hdf5_num_steps"])
        row["match_cost"] = float(mapped["match_cost"])
        row["alignment_cost"] = float(mapped.get("alignment_cost", mapped["match_cost"]))
        row["length_match"] = bool(mapped.get("length_match", row["num_steps"] == row["hdf5_num_steps"]))
        row["frame_indices"] = [int(index) for index in mapped.get("frame_indices", [])]
        row["source_hdf5"] = str(mapped["source_hdf5"])
        attached.append(row)
    return sorted(attached, key=lambda row: int(row["global_episode_index"]))


def attach_libero_task_ids(rows: list[dict[str, Any]], task_lookup: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row.setdefault("libero_benchmark_task_id", int(task_lookup[int(row["task_id"])]["libero_benchmark_task_id"]))
    return rows


def output_paths(output_dir: Path, task_id: int, demo_id: str, frame_id: int) -> dict[str, Path]:
    task_dir = f"task_{task_id:02d}"
    frame_name = f"{frame_id:06d}.json"
    return {
        "world": output_dir / "rule_based" / "world_graph" / task_dir / demo_id / frame_name,
        "observable": output_dir / "rule_based" / "observable_graph" / task_dir / demo_id / frame_name,
        "diagnostics": output_dir / "diagnostics" / task_dir / demo_id / frame_name,
        "frame": output_dir / "frames" / task_dir / demo_id / f"{frame_id:06d}.png",
    }


def demo_complete(output_dir: Path, row: dict[str, Any], expected_frames: int) -> bool:
    if expected_frames <= 0:
        return False
    last_frame = expected_frames - 1
    paths = output_paths(output_dir, int(row["task_id"]), str(row["demo_id"]), last_frame)
    return paths["world"].exists()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def generate_demo(
    *,
    output_dir: Path,
    row: dict[str, Any],
    task: dict[str, Any],
    config: dict[str, Any],
    max_frames: int | None,
    save_observable: bool,
    save_diagnostics: bool,
    save_frames: bool,
) -> dict[str, Any]:
    import h5py

    bddl_file = Path(task["bddl_file"])
    demo_file = Path(task["demo_file"])
    env = create_env(str(bddl_file), int(config["image_size"]))
    world_hashes: list[str] = []
    observable_hashes: list[str] = []
    records = 0
    started_at = time.time()
    try:
        with h5py.File(demo_file, "r") as f:
            demo_keys = sorted_demo_keys(f["data"])
            demo_index = int(row["libero_demo_index"])
            if demo_index >= len(demo_keys):
                raise IndexError(f"{demo_file} has {len(demo_keys)} demos, requested index {demo_index}")
            demo_key = demo_keys[demo_index]
            if demo_key != row.get("libero_demo_key"):
                raise ValueError(f"mapping expected {row.get('libero_demo_key')}, got {demo_key} in {demo_file}")
            group = f[f"data/{demo_key}"]
            states = group["states"][()]
            model_xml = h5_attr_text(group.attrs.get("model_file"))
            frame_indices = row.get("frame_indices") or list(range(len(states)))
            if max(frame_indices, default=-1) >= len(states):
                raise IndexError(f"frame_indices exceed HDF5 length for {row['demo_id']}: hdf5={len(states)}")
            limit = len(frame_indices) if max_frames is None else min(len(frame_indices), max_frames)
            expected_steps = int(row.get("num_steps", limit))
            if max_frames is None and len(frame_indices) != expected_steps:
                raise ValueError(
                    f"frame count mismatch for {row['demo_id']}: mapped={len(frame_indices)}, openvla={expected_steps}, "
                    f"demo={demo_key}, file={demo_file}"
                )
            for frame_id in range(limit):
                source_frame_id = int(frame_indices[frame_id])
                paths = output_paths(output_dir, int(row["task_id"]), str(row["demo_id"]), frame_id)
                obs = set_demo_state(env, states[source_frame_id], model_xml if frame_id == 0 else None)
                world_graph, observable_graph, diagnostics = build_frame_graphs(
                    env=env,
                    obs=obs,
                    bddl_file=bddl_file,
                    task_id=str(row["task_id"]),
                    demo_id=str(row["demo_id"]),
                    frame_id=frame_id,
                    config=config,
                )
                write_json(paths["world"], world_graph)
                if save_observable:
                    write_json(paths["observable"], observable_graph)
                if save_diagnostics:
                    diagnostics["source_hdf5_frame_id"] = source_frame_id
                    write_json(paths["diagnostics"], diagnostics)
                if save_frames:
                    save_rgb(obs, paths["frame"])
                world_hashes.append(sha256_payload(world_graph))
                observable_hashes.append(sha256_payload(observable_graph))
                records += 1
    finally:
        try:
            env.close()
        except Exception:
            pass

    return {
        "global_episode_index": int(row["global_episode_index"]),
        "task_id": int(row["task_id"]),
        "task_name": row["task_name"],
        "language_instruction": row["language_instruction"],
        "demo_id": row["demo_id"],
        "libero_demo_index": int(row["libero_demo_index"]),
        "libero_demo_key": row["libero_demo_key"],
        "expected_num_steps": int(row["num_steps"]),
        "hdf5_num_steps": int(row.get("hdf5_num_steps", row["num_steps"])),
        "length_match": bool(row.get("length_match", True)),
        "match_cost": float(row.get("match_cost", 0.0)),
        "alignment_cost": float(row.get("alignment_cost", 0.0)),
        "records_written": records,
        "world_hashes": world_hashes,
        "observable_hashes": observable_hashes,
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def main() -> None:
    args = parse_args()
    rows = read_counts_csv(args.full_counts_csv)
    mapping_path = args.mapping_jsonl or (args.output_dir / "reports" / "openvla_hdf5_mapping.jsonl")
    if mapping_path.exists():
        rows = attach_demo_indices_from_mapping(rows, mapping_path)
    elif args.allow_rank_demo_fallback:
        rows = attach_demo_indices(rows)
    else:
        raise FileNotFoundError(
            f"missing mapping file: {mapping_path}. Run scene_graph_gt_v0/scripts/map_openvla_spatial_rlds_to_hdf5.py first."
        )
    if args.limit_demos is not None:
        rows = rows[: args.limit_demos]

    task_lookup = build_task_lookup(args.suite, rows)
    rows = attach_libero_task_ids(rows, task_lookup)
    config = default_rule_config(image_size=args.image_size)
    config["dataset_name"] = "libero_spatial_no_noops"
    config["openvla_counts_csv"] = str(args.full_counts_csv)
    config["openvla_hdf5_mapping_jsonl"] = str(mapping_path)
    config["output_layout"] = "rule_based/{world_graph,observable_graph}/task_XX/global_YYYYYY/frame.json"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "reports" / "rule_config.json", config)
    write_jsonl(args.output_dir / "reports" / "openvla_demo_manifest.jsonl", rows)

    summaries: list[dict[str, Any]] = []
    total_frames = 0
    total_written = 0
    started_at = time.time()
    for index, row in enumerate(rows, start=1):
        expected_frames = int(row["num_steps"]) if args.max_frames is None else min(int(row["num_steps"]), args.max_frames)
        if args.resume and demo_complete(args.output_dir, row, expected_frames):
            summary = {
                "global_episode_index": int(row["global_episode_index"]),
                "task_id": int(row["task_id"]),
                "demo_id": row["demo_id"],
                "skipped": True,
                "expected_num_steps": int(row["num_steps"]),
                "records_written": 0,
            }
            summaries.append(summary)
            print(
                f"[{index}/{len(rows)}] skip {row['demo_id']} task={row['task_id']} frames={expected_frames}",
                flush=True,
            )
            continue

        print(
            f"[{index}/{len(rows)}] generate {row['demo_id']} task={row['task_id']} "
            f"demo={row['libero_demo_index']} frames={expected_frames}",
            flush=True,
        )
        summary = generate_demo(
            output_dir=args.output_dir,
            row=row,
            task=task_lookup[int(row["task_id"])],
            config=config,
            max_frames=args.max_frames,
            save_observable=args.save_observable,
            save_diagnostics=args.save_diagnostics,
            save_frames=args.save_frames,
        )
        summaries.append(summary)
        total_frames += expected_frames
        total_written += int(summary["records_written"])
        if index % 10 == 0 or index == len(rows):
            write_json(
                args.output_dir / "reports" / "generation_summary.json",
                {
                    "counts_csv": str(args.full_counts_csv),
                    "mapping_jsonl": str(mapping_path),
                    "suite": args.suite,
                    "output_dir": str(args.output_dir),
                    "demos_targeted": len(rows),
                    "demos_processed": index,
                    "frames_targeted_so_far": total_frames,
                    "frames_written_so_far": total_written,
                    "elapsed_sec": round(time.time() - started_at, 3),
                    "config": config,
                    "demo_summaries": summaries,
                },
            )

    write_json(
        args.output_dir / "reports" / "generation_summary.json",
        {
            "counts_csv": str(args.full_counts_csv),
            "mapping_jsonl": str(mapping_path),
            "suite": args.suite,
            "output_dir": str(args.output_dir),
            "demos_targeted": len(rows),
            "demos_processed": len(rows),
            "frames_targeted_so_far": total_frames,
            "frames_written_so_far": total_written,
            "elapsed_sec": round(time.time() - started_at, 3),
            "config": config,
            "demo_summaries": summaries,
        },
    )


if __name__ == "__main__":
    main()
