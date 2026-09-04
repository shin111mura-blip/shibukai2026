#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "openvla_rollout_graph_v2"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"
JOBS_FULL = DATA_ROOT / "manifests" / "jobs_full.jsonl"
SCHEMA_LOCK_JSON = DATA_ROOT / "graph_schema_lock.json"
POLICY_POOL_CONFIG = CONFIGS / "rollout_policy_pool.yaml"
COLLECTION_CONFIG = CONFIGS / "rollout_collection_libero_spatial_full.yaml"

ORACLE_SOURCE_FILES = [
    ROOT / "scene_graph_gt_v0" / "scene_graph" / "rule_generator.py",
    ROOT / "scene_graph_gt_v0" / "scene_graph" / "node_extractor.py",
    ROOT / "scene_graph_gt_v0" / "scene_graph" / "relation_rules.py",
    ROOT / "scene_graph_gt_v0" / "scene_graph" / "grasp_detector.py",
    ROOT / "scene_graph_gt_v0" / "scene_graph" / "canonicalize.py",
    ROOT / "scene_graph_gt_v0" / "scene_graph" / "schema.py",
]
TARGET_BUILDER_FILES = [
    ROOT / "scene_graph_generator" / "graph_generator" / "targets.py",
    ROOT / "scene_graph_generator" / "graph_generator" / "masks.py",
]
ONTOLOGY_PATH = ROOT / "outputs" / "scene_graph_generator_openvla_spatial" / "ontology" / "ontology.json"
WORLD_GRAPH_ROOT = ROOT / "outputs" / "scene_graph_gt_openvla_spatial" / "rule_based" / "world_graph"
TEACHER_GRAPH_3D_ROOT = ROOT / "outputs" / "scene_graph_generator_openvla_spatial" / "teacher_graph_3d" / "world_graph"
OPENVLA_DEMO_MANIFEST = ROOT / "outputs" / "scene_graph_gt_openvla_spatial" / "reports" / "openvla_demo_manifest.jsonl"
MAPPING_JSONL = ROOT / "outputs" / "scene_graph_gt_openvla_spatial" / "reports" / "openvla_hdf5_mapping.jsonl"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            f.write("\n")
    tmp.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(f"{path} is not JSON and PyYAML is unavailable") from exc
        return yaml.safe_load(text)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_payload(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def command_output(args: list[str], cwd: Path = ROOT) -> str:
    try:
        out = subprocess.check_output(args, cwd=str(cwd), stderr=subprocess.STDOUT, text=True, timeout=20)
        return out.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def git_commit(path: Path) -> str:
    return command_output(["git", "rev-parse", "HEAD"], cwd=path)


def find_checkpoint_candidates() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for adapter in ROOT.rglob("adapter_config.json"):
        if ".git" in adapter.parts:
            continue
        model_file = adapter.parent / "adapter_model.safetensors"
        meta_file = adapter.parent.parent / "checkpoint_metadata.json"
        run_meta = adapter.parent.parent.parent / "run_metadata.json"
        row = {
            "adapter_path": str(adapter.parent.relative_to(ROOT)),
            "adapter_config": str(adapter.relative_to(ROOT)),
            "adapter_model_exists": model_file.exists(),
            "adapter_model_sha256": sha256_file(model_file) if model_file.exists() else None,
            "checkpoint_metadata": str(meta_file.relative_to(ROOT)) if meta_file.exists() else None,
            "run_metadata": str(run_meta.relative_to(ROOT)) if run_meta.exists() else None,
            "condition": None,
            "seed": None,
            "training_data_percentage": None,
            "uses_depth": None,
            "uses_graph_aux": None,
        }
        for meta_path in [run_meta, meta_file]:
            if meta_path.exists():
                try:
                    meta = read_json(meta_path)
                except Exception:
                    continue
                row["condition"] = row["condition"] or meta.get("condition")
                row["seed"] = row["seed"] or meta.get("seed")
                row["uses_depth"] = meta.get("uses_depth", row["uses_depth"])
                row["uses_graph_aux"] = meta.get("uses_graph_aux", row["uses_graph_aux"])
                manifest = str(meta.get("manifest", ""))
                if "10pct" in manifest:
                    row["training_data_percentage"] = 10
                if "30pct" in manifest:
                    row["training_data_percentage"] = 30
        path_text = str(adapter.parent)
        if "rgb_action" in path_text:
            row["condition"] = row["condition"] or "rgb_action"
            row["uses_graph_aux"] = False if row["uses_graph_aux"] is None else row["uses_graph_aux"]
            row["uses_depth"] = False if row["uses_depth"] is None else row["uses_depth"]
        if "10pct" in path_text:
            row["training_data_percentage"] = row["training_data_percentage"] or 10
        if "30pct" in path_text:
            row["training_data_percentage"] = row["training_data_percentage"] or 30
        candidates.append(row)
    return {"adapter_candidates": sorted(candidates, key=lambda r: r["adapter_path"])}


def task_rows_from_manifest() -> dict[int, dict[str, Any]]:
    rows = read_jsonl(OPENVLA_DEMO_MANIFEST)
    tasks: dict[int, dict[str, Any]] = {}
    for row in rows:
        task_id = int(row.get("task_id", row.get("openvla_task_id")))
        tasks.setdefault(
            task_id,
            {
                "task_id": task_id,
                "task_name": row.get("task_name", f"task_{task_id:02d}"),
                "instruction": row.get("language_instruction") or row.get("instruction") or "",
            },
        )
    return tasks


def build_schema_lock() -> dict[str, Any]:
    ontology = read_json(ONTOLOGY_PATH) if ONTOLOGY_PATH.exists() else {"nodes": {}, "predicates": {}}
    num_nodes = len(ontology.get("nodes", {}))
    num_predicates = len(ontology.get("predicates", {}))
    node_order = [k for k, _v in sorted(ontology.get("nodes", {}).items(), key=lambda kv: kv[1]["index"])]
    relation_order = [k for k, v in sorted(ontology.get("predicates", {}).items(), key=lambda kv: kv[1])]
    relation_vocab_payload = {"label_to_id": ontology.get("predicates", {})}
    schema = {
        "dataset_schema_version": "openvla_rollout_graph_v2",
        "graph_schema_version": "oracle_rule_based_world_graph_v1",
        "created_from": "existing rule-based LIBERO/MuJoCo oracle graph code",
        "ORACLE_GRAPH_SOURCE_FILES": [file_record(p) for p in ORACLE_SOURCE_FILES],
        "ORACLE_GRAPH_ENTRYPOINT": "scene_graph_gt_v0/scene_graph/rule_generator.py",
        "ORACLE_GRAPH_FUNCTION_OR_CLASS": "build_frame_graphs(env, obs, bddl_file, task_id, demo_id, frame_id, config)",
        "GRAPH_TENSOR_BUILDER": [file_record(p) for p in TARGET_BUILDER_FILES],
        "RELATION_RULE_SOURCE_FILES": [file_record(ROOT / "scene_graph_gt_v0" / "scene_graph" / "relation_rules.py")],
        "CONTACT_RULE_SOURCE": file_record(ROOT / "scene_graph_gt_v0" / "scene_graph" / "node_extractor.py"),
        "GRASP_RULE_SOURCE": file_record(ROOT / "scene_graph_gt_v0" / "scene_graph" / "grasp_detector.py"),
        "TOUCHING_RULE_SOURCE": "touching is contact diagnostic only; forbidden as Graph relation in locked rule config",
        "ONTOLOGY_PATH": file_record(ONTOLOGY_PATH),
        "NODE_ORDER_PATH": str(ONTOLOGY_PATH.relative_to(ROOT)),
        "RELATION_ORDER_PATH": str(ONTOLOGY_PATH.relative_to(ROOT)),
        "GRAPH_SCHEMA_PATH": "scene_graph_generator/graph_generator/schema.py",
        "GRAPH_SCHEMA_SHA256": sha256_file(ROOT / "scene_graph_generator" / "graph_generator" / "schema.py"),
        "GRAPH_SCHEMA_VERSION": "dense_binary_targets_from_ontology_v1",
        "GRAPH_TARGET_SHAPE": [num_nodes, num_nodes, num_predicates],
        "NODE_MASK_SHAPE": [num_nodes],
        "RELATION_MASK_SHAPE": [num_nodes, num_nodes, num_predicates],
        "POSITION_TARGET_SHAPE": [num_nodes, 3],
        "ONTOLOGY_SHA256": sha256_file(ONTOLOGY_PATH) if ONTOLOGY_PATH.exists() else None,
        "RELATION_VOCABULARY_SHA256": sha256_payload(relation_vocab_payload),
        "node_order": node_order,
        "relation_order": relation_order,
        "forbidden_predicates": ["between", "touching", "near", "overlapping", "holding"],
        "positive_predicates": relation_order,
        "depth_used_for_policy": False,
        "depth_used_for_graph_generator_input": False,
        "oracle_uses_simulator_privileged_state": True,
    }
    return schema


def write_schema_reports(schema: dict[str, Any]) -> None:
    write_json(SCHEMA_LOCK_JSON, schema)
    lines = [
        "# Oracle Graph Schema Lock",
        "",
        f"- dataset_schema_version: `{schema['dataset_schema_version']}`",
        f"- graph_schema_version: `{schema['graph_schema_version']}`",
        f"- oracle entrypoint: `{schema['ORACLE_GRAPH_ENTRYPOINT']}`",
        f"- oracle function: `{schema['ORACLE_GRAPH_FUNCTION_OR_CLASS']}`",
        f"- tensor target shape: `{schema['GRAPH_TARGET_SHAPE']}`",
        f"- node mask shape: `{schema['NODE_MASK_SHAPE']}`",
        f"- relation mask shape: `{schema['RELATION_MASK_SHAPE']}`",
        f"- position target shape: `{schema['POSITION_TARGET_SHAPE']}`",
        f"- ontology sha256: `{schema['ONTOLOGY_SHA256']}`",
        f"- graph schema sha256: `{schema['GRAPH_SCHEMA_SHA256']}`",
        f"- relation vocabulary sha256: `{schema['RELATION_VOCABULARY_SHA256']}`",
        f"- nodes: `{', '.join(schema['node_order'])}`",
        f"- relations: `{', '.join(schema['relation_order'])}`",
        "",
        "Depth is not used for policy input or Graph Generator input. Oracle labels use simulator privileged state.",
    ]
    path = REPORTS / "oracle_graph_schema_lock.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_policies() -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    candidates = find_checkpoint_candidates()["adapter_candidates"]
    requested_policy_ids = {"high_official_libero_spatial", "low_10pct_action_only"}
    if COLLECTION_CONFIG.exists():
        try:
            collection_config = load_config(COLLECTION_CONFIG)
            requested_policy_ids = {
                str(item["policy_id"])
                for item in collection_config.get("preflight_jobs", [])
                if "policy_id" in item
            }
            requested_policy_ids.update(str(policy_id) for policy_id in collection_config.get("full_plan", {}))
        except Exception:
            pass
    base = ROOT / "checkpoints" / "openvla_7b_base_with_libero_spatial_10demo_stats"
    high_local = ROOT / "checkpoints" / "openvla_7b_finetuned_libero_spatial"
    high_bound = ROOT / "checkpoints" / "openvla_libero_spatial_ft"
    high_path = high_local if high_local.exists() else high_bound
    if "high_official_libero_spatial" in requested_policy_ids and (
        not high_path.exists() or not any(high_path.iterdir())
    ):
        blockers.append("High official LIBERO-Spatial checkpoint is not present locally; current eval code and Docker are offline/local_files_only.")
    low_candidates = [
        c
        for c in candidates
        if c.get("condition") == "rgb_action"
        and c.get("uses_graph_aux") is False
        and c.get("uses_depth") is False
        and c.get("adapter_model_exists")
        and (c.get("training_data_percentage") == 10 or "server_rgb_action_1k" in c["adapter_path"])
    ]
    middle_candidates = [
        c
        for c in candidates
        if c.get("condition") == "rgb_action"
        and c.get("uses_graph_aux") is False
        and c.get("uses_depth") is False
        and c.get("adapter_model_exists")
        and c.get("training_data_percentage") == 30
    ]
    if "low_10pct_action_only" in requested_policy_ids and not low_candidates:
        blockers.append("Low 10pct action-only checkpoint was not found as a fully identified local checkpoint.")
    if "middle_30pct_action_only" in requested_policy_ids and not middle_candidates:
        blockers.append("Middle 30pct action-only checkpoint was not found as a fully identified local checkpoint.")
    low = sorted(low_candidates, key=lambda c: (str(c.get("seed")), c["adapter_path"]))[0] if low_candidates else None
    middle = sorted(middle_candidates, key=lambda c: (str(c.get("seed")), c["adapter_path"]))[0] if middle_candidates else None
    stats_path = base / "dataset_statistics.json"
    pool = {
        "policies": {
            "high_official_libero_spatial": {
                "policy_id": "high_official_libero_spatial",
                "model_id": "openvla/openvla-7b-finetuned-libero-spatial",
                "revision": "main",
                "local_checkpoint_path": str(high_path.relative_to(ROOT)) if high_path.exists() else None,
                "checkpoint_sha256": None,
                "dataset_statistics_path": None,
                "dataset_statistics_sha256": None,
                "unnorm_key": "libero_spatial",
                "center_crop": True,
                "prompt_template_id": "openvla_default_or_openvla_v01_by_checkpoint_name",
                "enabled": high_path.exists() and any(high_path.iterdir()),
            },
            "middle_30pct_action_only": {
                "policy_id": "middle_30pct_action_only",
                "checkpoint_path": middle["adapter_path"] if middle else None,
                "checkpoint_sha256": middle["adapter_model_sha256"] if middle else None,
                "training_data_percentage": 30,
                "dataset_statistics_path": str(stats_path.relative_to(ROOT)) if stats_path.exists() else None,
                "dataset_statistics_sha256": sha256_file(stats_path) if stats_path.exists() else None,
                "unnorm_key": "libero_spatial_no_noops",
                "center_crop": True,
                "prompt_template_id": "openvla_default_or_openvla_v01_by_checkpoint_name",
                "enabled": middle is not None,
            },
            "low_10pct_action_only": {
                "policy_id": "low_10pct_action_only",
                "checkpoint_path": low["adapter_path"] if low else None,
                "checkpoint_sha256": low["adapter_model_sha256"] if low else None,
                "training_data_percentage": 10,
                "dataset_statistics_path": str(stats_path.relative_to(ROOT)) if stats_path.exists() else None,
                "dataset_statistics_sha256": sha256_file(stats_path) if stats_path.exists() else None,
                "unnorm_key": "libero_spatial_no_noops",
                "center_crop": True,
                "prompt_template_id": "openvla_default_or_openvla_v01_by_checkpoint_name",
                "enabled": low is not None,
            },
        },
        "selection_blockers": blockers,
        "candidate_inventory": candidates,
    }
    return pool, blockers


def write_policy_reports(pool: dict[str, Any]) -> None:
    write_json(POLICY_POOL_CONFIG, pool)
    lines = ["# Rollout Policy Selection", ""]
    for policy in pool["policies"].values():
        lines.extend(
            [
                f"## {policy['policy_id']}",
                "",
                f"- enabled: `{policy.get('enabled')}`",
                f"- model/checkpoint: `{policy.get('model_id') or policy.get('checkpoint_path') or policy.get('local_checkpoint_path')}`",
                f"- checkpoint sha256: `{policy.get('checkpoint_sha256')}`",
                f"- dataset statistics: `{policy.get('dataset_statistics_path')}`",
                f"- dataset statistics sha256: `{policy.get('dataset_statistics_sha256')}`",
                f"- unnorm_key: `{policy.get('unnorm_key')}`",
                f"- center_crop: `{policy.get('center_crop')}`",
                "",
            ]
        )
    if pool["selection_blockers"]:
        lines.append("## Critical Blockers")
        lines.extend(f"- {b}" for b in pool["selection_blockers"])
    path = REPORTS / "rollout_policy_selection.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_collection_config() -> dict[str, Any]:
    return {
        "suite": "libero_spatial",
        "output_dir": "data/openvla_rollout_graph_v2",
        "task_ids": list(range(10)),
        "maximum_steps": 220,
        "num_steps_wait": 10,
        "image_size": 256,
        "camera_name": "agentview",
        "save_terminal_frame": True,
        "validate_after_episode": True,
        "split_seed": 20260803,
        "preflight_jobs": [
            {"policy_id": "high_official_libero_spatial", "task_id": 0, "assigned_worker": 0, "assigned_gpu": 0},
            {"policy_id": "high_official_libero_spatial", "task_id": 1, "assigned_worker": 1, "assigned_gpu": 1},
            {"policy_id": "low_10pct_action_only", "task_id": 2, "assigned_worker": 2, "assigned_gpu": 2},
            {"policy_id": "low_10pct_action_only", "task_id": 3, "assigned_worker": 3, "assigned_gpu": 3},
        ],
        "full_plan": {
            "high_official_libero_spatial": {
                "episodes_per_task": 40,
                "assigned_workers": [0, 1],
                "assigned_gpus": [0, 1],
            },
            "low_10pct_action_only": {
                "episodes_per_task": 40,
                "assigned_workers": [2, 3],
                "assigned_gpus": [2, 3],
            },
        },
        "additional_collection": {
            "min_success_per_task": 30,
            "min_failure_per_task": 30,
            "max_episodes_per_task": 120,
            "success_shortage_policy_priority": ["high_official_libero_spatial"],
            "failure_shortage_policy_priority": ["low_10pct_action_only"],
        },
    }


def write_default_configs() -> None:
    if not COLLECTION_CONFIG.exists():
        write_json(COLLECTION_CONFIG, default_collection_config())


def deterministic_job_id(policy_id: str, task_id: int, initial_state_id: int, rollout_seed: int) -> str:
    raw = f"{policy_id}|task_{task_id:02d}|init_{initial_state_id:03d}|seed_{rollout_seed}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def make_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = task_rows_from_manifest()
    jobs: list[dict[str, Any]] = []
    seed_base = 2026080300
    for item in config["preflight_jobs"]:
        task_id = int(item["task_id"])
        seed = seed_base + len(jobs)
        policy_id = item["policy_id"]
        job_id = deterministic_job_id(policy_id, task_id, 0, seed)
        jobs.append(
            {
                "job_id": job_id,
                "phase": "preflight",
                "policy_id": policy_id,
                "task_id": task_id,
                "task_name": tasks.get(task_id, {}).get("task_name", f"task_{task_id:02d}"),
                "rollout_index": 0,
                "initial_state_id": 0,
                "rollout_seed": seed,
                "assigned_worker": int(item["assigned_worker"]),
                "assigned_gpu": int(item["assigned_gpu"]),
                "output_path": f"data/openvla_rollout_graph_v2/episodes/{policy_id}/worker_{int(item['assigned_worker']):02d}/{job_id}",
                "status": "pending",
            }
        )
    for policy_id, plan in config["full_plan"].items():
        for task_id in config["task_ids"]:
            count = int(plan["episodes_per_task"])
            for rollout_index in range(count):
                if "assigned_workers" in plan:
                    worker = int(plan["assigned_workers"][rollout_index % len(plan["assigned_workers"])])
                    gpu = int(plan["assigned_gpus"][rollout_index % len(plan["assigned_gpus"])])
                else:
                    worker = int(plan["assigned_worker"])
                    gpu = int(plan["assigned_gpu"])
                initial_state_id = rollout_index
                seed = seed_base + 10000 + len(jobs)
                job_id = deterministic_job_id(policy_id, int(task_id), initial_state_id, seed)
                jobs.append(
                    {
                        "job_id": job_id,
                        "phase": "full",
                        "policy_id": policy_id,
                        "task_id": int(task_id),
                        "task_name": tasks.get(int(task_id), {}).get("task_name", f"task_{int(task_id):02d}"),
                        "rollout_index": rollout_index,
                        "initial_state_id": initial_state_id,
                        "rollout_seed": seed,
                        "assigned_worker": worker,
                        "assigned_gpu": gpu,
                        "output_path": f"data/openvla_rollout_graph_v2/episodes/{policy_id}/worker_{worker:02d}/{job_id}",
                        "status": "pending",
                    }
                )
    return jobs


def validate_episode_dir(episode_dir: Path, schema: dict[str, Any] | None = None) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    metadata_path = episode_dir / "metadata.json"
    arrays_path = episode_dir / "frames.npz"
    complete_path = episode_dir / "COMPLETE"
    checksum_path = episode_dir / "checksums.sha256"
    if not metadata_path.exists():
        errors.append("metadata.json missing")
    if not arrays_path.exists():
        errors.append("frames.npz missing")
    if not complete_path.exists():
        errors.append("COMPLETE marker missing")
    if not checksum_path.exists():
        errors.append("checksums.sha256 missing")
    meta: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            meta = read_json(metadata_path)
        except Exception as exc:
            errors.append(f"metadata JSON invalid: {exc}")
    if arrays_path.exists():
        try:
            data = np.load(arrays_path, allow_pickle=True)
            if "rgb" in data:
                rgb = data["rgb"]
                if rgb.dtype != np.uint8:
                    errors.append(f"rgb dtype is {rgb.dtype}, expected uint8")
                if rgb.ndim != 4 or rgb.shape[-1] != 3:
                    errors.append(f"rgb shape is {list(rgb.shape)}, expected T,H,W,3")
            else:
                errors.append("rgb array missing")
            if "oracle_graph_tensor" in data and schema:
                expected = tuple(schema["GRAPH_TARGET_SHAPE"])
                got = tuple(data["oracle_graph_tensor"].shape[1:])
                if got != expected:
                    errors.append(f"graph tensor shape {got} != {expected}")
            for key in data.files:
                arr = data[key]
                if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                    errors.append(f"{key} contains NaN or Inf")
        except Exception as exc:
            errors.append(f"frames.npz invalid: {type(exc).__name__}: {exc}")
    return not errors, errors, meta


def atomic_write_episode(final_dir: Path, writer) -> Path:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=final_dir.name + ".", dir=str(final_dir.parent)))
    try:
        writer(tmp_dir)
        if final_dir.exists():
            if (final_dir / "COMPLETE").exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return final_dir
            quarantine_root = final_dir.parents[3] / "quarantine_existing_episode_dirs"
            quarantine_root.mkdir(parents=True, exist_ok=True)
            quarantine_dir = quarantine_root / f"{final_dir.parent.name}_{final_dir.name}_{int(time.time())}"
            shutil.move(str(final_dir), str(quarantine_dir))
        os.replace(tmp_dir, final_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return final_dir


def add_runtime_paths() -> None:
    for p in [ROOT / "openvla", ROOT / "LIBERO", ROOT / "scene_graph_gt_v0", ROOT]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


@dataclass(frozen=True)
class PolicyRuntime:
    policy_id: str
    checkpoint: Path
    unnorm_key: str
    center_crop: bool


def resolve_policy_runtime(policy_pool: dict[str, Any], policy_id: str) -> PolicyRuntime:
    policy = policy_pool["policies"][policy_id]
    if not policy.get("enabled"):
        raise RuntimeError(f"policy {policy_id} is disabled or unresolved")
    checkpoint = policy.get("checkpoint_path") or policy.get("local_checkpoint_path")
    if checkpoint is None:
        raise RuntimeError(f"policy {policy_id} has no local checkpoint path")
    checkpoint_path = ROOT / checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint path not found: {checkpoint_path}")
    return PolicyRuntime(
        policy_id=policy_id,
        checkpoint=checkpoint_path,
        unnorm_key=str(policy.get("unnorm_key") or "libero_spatial"),
        center_crop=bool(policy.get("center_crop", True)),
    )


def encode_graph_targets(graph: dict[str, Any], ontology: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    add_runtime_paths()
    from scene_graph_generator.graph_generator.masks import relation_validity_mask
    from scene_graph_generator.graph_generator.targets import encode_targets

    y_node, y_edge = encode_targets(graph, ontology)
    relation_mask = relation_validity_mask(ontology)
    return y_node.astype(np.float32), y_edge.astype(np.float32), relation_mask.astype(bool)


def classify_failure(success: bool, terminal_reason: str, frames: list[dict[str, Any]]) -> str:
    if success:
        return "success"
    if terminal_reason == "timeout":
        return "timeout"
    ever_contact = any(frame.get("contact_count", 0) > 0 for frame in frames)
    ever_grasp = any(frame.get("has_grasping", False) for frame in frames)
    if not ever_contact:
        return "no_contact"
    if ever_contact and not ever_grasp:
        return "failed_grasp"
    return "unknown"


def checksum_episode_files(episode_dir: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(p for p in episode_dir.rglob("*") if p.is_file() and p.name != "checksums.sha256"):
        checksums[str(path.relative_to(episode_dir))] = sha256_file(path)
    with (episode_dir / "checksums.sha256").open("w", encoding="utf-8") as f:
        for rel, digest in checksums.items():
            f.write(f"{digest}  {rel}\n")
    return checksums


def main_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=COLLECTION_CONFIG)
    return parser
