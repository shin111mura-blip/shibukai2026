#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap

bootstrap()

from scene_graph.rule_generator import resolve_task, sorted_demo_keys


DEFAULT_COUNTS_CSV = Path(
    "outputs/openvla_libero_spatial_lora_pct_sweep_subsetseed42_rerun/dataset_audit/full_counts.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/scene_graph_gt_openvla_spatial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map OpenVLA LIBERO-Spatial RLDS global episodes to local LIBERO HDF5 demo indices."
    )
    parser.add_argument("--data-root-dir", type=Path, default=Path("data/modified_libero_rlds"))
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--split", default="train")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--full-counts-csv", type=Path, default=DEFAULT_COUNTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-cost-warning", type=float, default=1e-3)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def read_counts(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            global_index = int(row["global_episode_index"])
            rows[global_index] = {
                "global_episode_index": global_index,
                "openvla_task_id": int(row["task_id"]),
                "task_name": row["task_name"],
                "language_instruction": row["language_instruction"],
                "num_steps": int(row["num_steps"]),
                "episode_length": int(row.get("episode_length") or row["num_steps"]),
                "episode_id": row.get("episode_id", ""),
            }
    return rows


def decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        return decode_text(value.item())
    return str(value)


def load_rlds_episodes(args: argparse.Namespace, counts: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np
    import tensorflow_datasets as tfds

    builder = tfds.builder(args.dataset_name, data_dir=str(args.data_root_dir))
    dataset = builder.as_dataset(split=args.split, shuffle_files=False)
    episodes: list[dict[str, Any]] = []
    for global_index, episode in enumerate(tfds.as_numpy(dataset)):
        steps = episode["steps"]
        if isinstance(steps, dict):
            joint = np.asarray(steps["observation"]["joint_state"], dtype=float)
            state = np.asarray(steps["observation"]["state"], dtype=float)
            language = decode_text(steps["language_instruction"][0])
        else:
            step_list = list(steps)
            joint = np.stack([step["observation"]["joint_state"] for step in step_list]).astype(float)
            state = np.stack([step["observation"]["state"] for step in step_list]).astype(float)
            language = decode_text(step_list[0]["language_instruction"])
        row = dict(counts.get(global_index, {}))
        if row and row["language_instruction"] != language:
            raise ValueError(
                f"language mismatch at global episode {global_index}: "
                f"counts={row['language_instruction']!r}, rlds={language!r}"
            )
        row.update(
            {
                "global_episode_index": global_index,
                "language_instruction": language,
                "joint_state": joint,
                "eef_gripper_state": state,
                "num_steps": len(joint),
            }
        )
        episodes.append(row)
    return episodes


def load_hdf5_demos(suite: str, language: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import h5py
    import numpy as np

    task = resolve_task(suite, language)
    demos: list[dict[str, Any]] = []
    with h5py.File(task["demo_file"], "r") as h5:
        keys = sorted_demo_keys(h5["data"])
        for demo_index, key in enumerate(keys):
            group = h5[f"data/{key}"]
            joint = np.asarray(group["obs/joint_states"], dtype=float)
            state = np.concatenate(
                [np.asarray(group["obs/ee_states"], dtype=float), np.asarray(group["obs/gripper_states"], dtype=float)],
                axis=1,
            )
            demos.append(
                {
                    "libero_demo_index": demo_index,
                    "libero_demo_key": key,
                    "num_steps": len(joint),
                    "joint_state": joint,
                    "eef_gripper_state": state,
                }
            )
    return task, demos


def sequence_cost(episode: dict[str, Any], demo: dict[str, Any]) -> float:
    import numpy as np

    ep_joint = episode["joint_state"]
    ep_state = episode["eef_gripper_state"]
    demo_joint = demo["joint_state"]
    demo_state = demo["eef_gripper_state"]
    length = min(len(ep_joint), len(demo_joint))
    length_penalty = abs(len(ep_joint) - len(demo_joint)) * 1000.0
    if length == 0:
        return float("inf")
    return float(
        length_penalty
        + np.mean(np.linalg.norm(ep_joint[:length] - demo_joint[:length], axis=1))
        + np.mean(np.linalg.norm(ep_state[:length] - demo_state[:length], axis=1))
    )


def frame_cost_matrix(episode: dict[str, Any], demo: dict[str, Any]):
    import numpy as np

    ep_joint = episode["joint_state"]
    ep_state = episode["eef_gripper_state"]
    demo_joint = demo["joint_state"]
    demo_state = demo["eef_gripper_state"]
    cost = np.linalg.norm(ep_joint[:, None, :] - demo_joint[None, :, :], axis=2)
    cost += np.linalg.norm(ep_state[:, None, :] - demo_state[None, :, :], axis=2)
    n, m = cost.shape
    if n > 1 and m > 1:
        ep_t = np.linspace(0.0, 1.0, n)[:, None]
        demo_t = np.linspace(0.0, 1.0, m)[None, :]
        cost += 0.01 * np.abs(ep_t - demo_t)
    return cost


def align_frame_indices(episode: dict[str, Any], demo: dict[str, Any]) -> tuple[list[int], float]:
    import numpy as np

    n = len(episode["joint_state"])
    m = len(demo["joint_state"])
    if n == m:
        indices = list(range(n))
        cost = frame_cost_matrix(episode, demo)
        return indices, float(np.mean(cost[np.arange(n), np.arange(n)]))

    cost = frame_cost_matrix(episode, demo)
    dp = np.full((n, m), np.inf, dtype=float)
    prev = np.full((n, m), -1, dtype=int)
    dp[0] = cost[0]
    for i in range(1, n):
        best_value = np.inf
        best_index = -1
        for j in range(m):
            if dp[i - 1, j] < best_value:
                best_value = dp[i - 1, j]
                best_index = j
            dp[i, j] = cost[i, j] + best_value
            prev[i, j] = best_index

    end = int(np.argmin(dp[-1]))
    indices = [end]
    for i in range(n - 1, 0, -1):
        end = int(prev[i, end])
        indices.append(end)
    indices.reverse()
    return indices, float(dp[-1, indices[-1]] / max(n, 1))


def match_task_episodes(
    *,
    suite: str,
    language: str,
    episodes: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task, demos = load_hdf5_demos(suite, language)
    pair_costs: list[tuple[float, int, int]] = []
    for episode_index, episode in enumerate(episodes):
        for demo_index, demo in enumerate(demos):
            if len(episode["joint_state"]) != len(demo["joint_state"]):
                continue
            pair_costs.append((sequence_cost(episode, demo), episode_index, demo_index))
    if len(pair_costs) == 0:
        for episode_index, episode in enumerate(episodes):
            for demo_index, demo in enumerate(demos):
                pair_costs.append((sequence_cost(episode, demo), episode_index, demo_index))

    assigned_episodes: set[int] = set()
    assigned_demos: set[int] = set()
    matches: list[dict[str, Any]] = []
    for cost, episode_index, demo_index in sorted(pair_costs):
        if episode_index in assigned_episodes or demo_index in assigned_demos:
            continue
        episode = episodes[episode_index]
        demo = demos[demo_index]
        assigned_episodes.add(episode_index)
        assigned_demos.add(demo_index)
        matches.append(
            {
                "global_episode_index": int(episode["global_episode_index"]),
                "openvla_task_id": int(episode["openvla_task_id"]),
                "libero_benchmark_task_id": int(task["task_id"]),
                "task_name": episode.get("task_name") or language,
                "language_instruction": language,
                "num_steps": int(episode["num_steps"]),
                "episode_length": int(episode.get("episode_length", episode["num_steps"])),
                "episode_id": episode.get("episode_id", ""),
                "source_hdf5": str(task["demo_file"]),
                "libero_demo_index": int(demo["libero_demo_index"]),
                "libero_demo_key": demo["libero_demo_key"],
                "hdf5_num_steps": int(demo["num_steps"]),
                "match_cost": cost,
                "length_match": int(episode["num_steps"]) == int(demo["num_steps"]),
                "alignment_cost": align_frame_indices(episode, demo)[1],
                "frame_indices": align_frame_indices(episode, demo)[0],
                "demo_id": f"global_{int(episode['global_episode_index']):06d}",
            }
        )

    if len(matches) != len(episodes):
        fallback_costs: list[tuple[float, int, int]] = []
        for episode_index, episode in enumerate(episodes):
            if episode_index in assigned_episodes:
                continue
            for demo_index, demo in enumerate(demos):
                if demo_index in assigned_demos:
                    continue
                fallback_costs.append((sequence_cost(episode, demo), episode_index, demo_index))
        for cost, episode_index, demo_index in sorted(fallback_costs):
            if episode_index in assigned_episodes or demo_index in assigned_demos:
                continue
            episode = episodes[episode_index]
            demo = demos[demo_index]
            assigned_episodes.add(episode_index)
            assigned_demos.add(demo_index)
            matches.append(
                {
                    "global_episode_index": int(episode["global_episode_index"]),
                    "openvla_task_id": int(episode["openvla_task_id"]),
                    "libero_benchmark_task_id": int(task["task_id"]),
                    "task_name": episode.get("task_name") or language,
                    "language_instruction": language,
                    "num_steps": int(episode["num_steps"]),
                    "episode_length": int(episode.get("episode_length", episode["num_steps"])),
                    "episode_id": episode.get("episode_id", ""),
                    "source_hdf5": str(task["demo_file"]),
                    "libero_demo_index": int(demo["libero_demo_index"]),
                    "libero_demo_key": demo["libero_demo_key"],
                    "hdf5_num_steps": int(demo["num_steps"]),
                    "match_cost": cost,
                    "length_match": int(episode["num_steps"]) == int(demo["num_steps"]),
                    "alignment_cost": align_frame_indices(episode, demo)[1],
                    "frame_indices": align_frame_indices(episode, demo)[0],
                    "demo_id": f"global_{int(episode['global_episode_index']):06d}",
                }
            )

    if len(matches) != len(episodes):
        missing = sorted(
            int(episodes[index]["global_episode_index"])
            for index in range(len(episodes))
            if index not in assigned_episodes
        )
        raise ValueError(f"failed to match {len(missing)} episodes for {language!r}: {missing[:10]}")
    return task, sorted(matches, key=lambda row: int(row["global_episode_index"]))


def strip_arrays(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"joint_state", "eef_gripper_state"}}


def main() -> None:
    args = parse_args()
    counts = read_counts(args.full_counts_csv)
    episodes = load_rlds_episodes(args, counts)
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_language[episode["language_instruction"]].append(episode)

    all_matches: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    for language in sorted(by_language):
        task, matches = match_task_episodes(
            suite=args.suite,
            language=language,
            episodes=sorted(by_language[language], key=lambda row: int(row["global_episode_index"])),
        )
        all_matches.extend(matches)
        max_cost = max(float(row["match_cost"]) for row in matches)
        max_alignment_cost = max(float(row["alignment_cost"]) for row in matches)
        length_mismatches = sum(1 for row in matches if not row["length_match"])
        task_summaries.append(
            {
                "language_instruction": language,
                "openvla_task_ids": sorted({int(row["openvla_task_id"]) for row in matches}),
                "libero_benchmark_task_id": int(task["task_id"]),
                "matched_demos": len(matches),
                "max_match_cost": max_cost,
                "max_alignment_cost": max_alignment_cost,
                "length_mismatches": length_mismatches,
                "cost_warning": max_alignment_cost > args.max_cost_warning,
            }
        )
        print(
            f"matched {len(matches):3d} episodes: openvla_task={task_summaries[-1]['openvla_task_ids']} "
            f"libero_task={task['task_id']} max_align={max_alignment_cost:.6g} "
            f"length_mismatch={length_mismatches} {language}",
            flush=True,
        )

    all_matches = sorted(all_matches, key=lambda row: int(row["global_episode_index"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "reports" / "openvla_hdf5_mapping.jsonl", all_matches)
    write_json(
        args.output_dir / "reports" / "openvla_hdf5_mapping_summary.json",
        {
            "dataset_name": args.dataset_name,
            "data_root_dir": str(args.data_root_dir),
            "split": args.split,
            "suite": args.suite,
            "full_counts_csv": str(args.full_counts_csv),
            "episodes": len(all_matches),
            "tasks": task_summaries,
        },
    )
    write_jsonl(args.output_dir / "reports" / "openvla_demo_manifest.jsonl", [strip_arrays(row) for row in all_matches])


if __name__ == "__main__":
    main()
