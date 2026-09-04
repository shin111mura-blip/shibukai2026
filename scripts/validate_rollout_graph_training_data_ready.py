#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rollout_xyz_targets import RolloutXyzTargetCache


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def resolve_episode_dir(value: str, data_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        if path.exists():
            return path
        parts = path.parts
        if data_root.name in parts:
            suffix = Path(*parts[parts.index(data_root.name) + 1 :])
            candidate = data_root / suffix
            if candidate.exists():
                return candidate
        return path
    candidate = data_root / path
    if candidate.exists():
        return candidate
    candidate = data_root.parent.parent / path
    if candidate.exists():
        return candidate
    return path


def count_sidecar_xyz(sidecar: Path, expected_frames: int) -> tuple[int, int, list[str]]:
    errors: list[str] = []
    payload = read_json(sidecar)
    records = payload.get("position_records") or []
    if len(records) != expected_frames:
        errors.append(f"sidecar frame count mismatch: records={len(records)} expected={expected_frames}")
    valid_frames = 0
    valid_points = 0
    for rec in records[:expected_frames]:
        if "xyz_mask_sum" in rec:
            n = int(rec.get("xyz_mask_sum") or 0)
        else:
            n = len(rec.get("world_positions") or {})
        valid_points += n
        valid_frames += int(n > 0)
    return valid_frames, valid_points, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate rollout RGB, teacher graph tensors, and Graph3D sidecars for local retraining.")
    parser.add_argument("--data-root", type=Path, default=Path("data/openvla_rollout_graph_v2"))
    parser.add_argument("--ontology", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/ontology/ontology.json"))
    parser.add_argument("--xyz-sidecar-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--require-xyz", action="store_true")
    parser.add_argument("--max-errors", type=int, default=50)
    args = parser.parse_args()

    data_root = args.data_root
    manifest = args.manifest or (data_root / "manifests" / "all.jsonl")
    sidecar_root = args.xyz_sidecar_root or (data_root / "inspection" / "graph3d_positions_all")
    ontology = read_json(args.ontology)
    xyz_cache = RolloutXyzTargetCache(ontology, data_root=data_root, sidecar_root=sidecar_root)

    rows = read_jsonl(manifest)
    errors: list[dict[str, Any]] = []
    totals = {
        "manifest_rows": len(rows),
        "episodes_checked": 0,
        "rgb_frames": 0,
        "teacher_graph_frames": 0,
        "teacher_graph_positive_edges": 0,
        "episodes_with_xyz": 0,
        "episodes_missing_xyz": 0,
        "xyz_valid_frames": 0,
        "xyz_valid_points": 0,
        "frames_npz_xyz_valid_points": 0,
        "episodes_with_errors": 0,
    }

    for row in rows:
        episode_dir = resolve_episode_dir(str(row["episode_dir"]), data_root)
        row_errors: list[str] = []
        if not episode_dir.exists():
            row_errors.append(f"episode missing: {episode_dir}")
            expected_frames = 0
        else:
            totals["episodes_checked"] += 1
            meta_path = episode_dir / "metadata.json"
            frames_path = episode_dir / "frames.npz"
            if not meta_path.exists():
                row_errors.append("metadata.json missing")
                expected_frames = 0
            else:
                meta = read_json(meta_path)
                expected_frames = int(meta.get("episode_length", 0))
            if not frames_path.exists():
                row_errors.append("frames.npz missing")
            else:
                arrays = np.load(frames_path)
                required = {
                    "rgb",
                    "oracle_graph_tensor",
                    "node_valid_mask",
                    "relation_valid_mask",
                    "position_target",
                    "position_valid_mask",
                    "executed_action",
                    "reward",
                    "done",
                }
                missing = sorted(required - set(arrays.files))
                if missing:
                    row_errors.append(f"frames.npz missing keys: {missing}")
                else:
                    rgb = arrays["rgb"]
                    graph = arrays["oracle_graph_tensor"]
                    pos_mask = arrays["position_valid_mask"]
                    if rgb.ndim != 4 or rgb.shape[-1] != 3:
                        row_errors.append(f"rgb shape invalid: {rgb.shape}")
                    if expected_frames and int(rgb.shape[0]) != expected_frames:
                        row_errors.append(f"rgb frame count mismatch: {rgb.shape[0]} expected={expected_frames}")
                    if expected_frames and int(graph.shape[0]) != expected_frames:
                        row_errors.append(f"oracle_graph_tensor frame count mismatch: {graph.shape[0]} expected={expected_frames}")
                    totals["rgb_frames"] += int(rgb.shape[0])
                    totals["teacher_graph_frames"] += int(graph.shape[0])
                    totals["teacher_graph_positive_edges"] += int(graph.sum())
                    totals["frames_npz_xyz_valid_points"] += int(pos_mask.sum())
        sidecar = xyz_cache.sidecar_path_for_episode(episode_dir)
        if sidecar is None or not sidecar.exists():
            totals["episodes_missing_xyz"] += 1
            row_errors.append("graph3d_positions.json sidecar missing")
        else:
            totals["episodes_with_xyz"] += 1
            valid_frames, valid_points, sidecar_errors = count_sidecar_xyz(sidecar, expected_frames)
            totals["xyz_valid_frames"] += valid_frames
            totals["xyz_valid_points"] += valid_points
            row_errors.extend(sidecar_errors)

        if row_errors:
            totals["episodes_with_errors"] += 1
        if row_errors and len(errors) < args.max_errors:
            errors.append(
                {
                    "episode_dir": str(episode_dir),
                    "policy_id": row.get("policy_id"),
                    "task_id": row.get("task_id"),
                    "episode_success": row.get("episode_success"),
                    "errors": row_errors,
                }
            )

    split_counts = {}
    for split in ("train", "validation", "test"):
        path = data_root / "manifests" / f"{split}.jsonl"
        if path.exists():
            split_counts[split] = len(read_jsonl(path))

    status = "ok"
    blocking: list[str] = []
    if not rows:
        blocking.append(f"manifest is empty: {manifest}")
    if args.expected_episodes is not None and len(rows) != args.expected_episodes:
        blocking.append(f"expected {args.expected_episodes} episodes but manifest has {len(rows)}")
    if totals["teacher_graph_positive_edges"] <= 0:
        blocking.append("teacher graph tensors contain no positive edges")
    if args.require_xyz and totals["xyz_valid_points"] <= 0:
        blocking.append("no valid Graph3D XYZ sidecar points found")
    if totals["episodes_with_errors"]:
        blocking.append(f"episode validation errors found: {totals['episodes_with_errors']} shown up to {args.max_errors}")
    if blocking:
        status = "failed"

    report = {
        "status": status,
        "data_root": str(data_root),
        "manifest": str(manifest),
        "xyz_sidecar_root": str(sidecar_root),
        "split_counts": split_counts,
        "totals": totals,
        "blocking": blocking,
        "errors": errors,
    }
    out = data_root / "reports" / "local_graph_training_data_ready.json"
    write_json(out, report)
    print(json.dumps({"status": status, "report": str(out), "totals": totals, "blocking": blocking}, sort_keys=True))
    raise SystemExit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
