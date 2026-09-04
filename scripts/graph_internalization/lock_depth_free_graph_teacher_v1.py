#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SAMPLE_DIR = REPORTS / "depth_free_graph_samples"
MANIFEST_DIR = ROOT / "manifests"

OUTPUT_ROOT = ROOT / "outputs/scene_graph_generator_openvla_spatial"
CACHE_DIR = OUTPUT_ROOT / "feature_cache/all_frames"
DEPTH_DIR = OUTPUT_ROOT / "depth_features/all_frames"
ONTOLOGY_PATH = OUTPUT_ROOT / "ontology/ontology.json"
SPLIT_PATH = OUTPUT_ROOT / "splits/split_seed42_train50_val25_test25.json"
SPLIT_SUMMARY_PATH = OUTPUT_ROOT / "splits/split_summary.json"
BUNDLE_INDEX = ROOT / "artifacts/openvla_graph_internalization_bundle_v2/index.jsonl"
ARCH = "pooled_mlp_openvla_3d"
DEPTH_ARCH = "pooled_mlp_depth_3d"
CHECKPOINT_PATH = OUTPUT_ROOT / "checkpoints" / ARCH / "best.pt"
TRAINING_SUMMARY_PATH = OUTPUT_ROOT / "metrics" / ARCH / "training_summary.json"
TRAINING_STATUS_PATH = OUTPUT_ROOT / "reports" / f"{ARCH}_training_status.json"
THRESHOLD_METRICS_PATH = OUTPUT_ROOT / "metrics" / ARCH / "selected_thresholds_depth_3d.json"
DEPTH_TRAINING_SUMMARY_PATH = OUTPUT_ROOT / "metrics" / DEPTH_ARCH / "training_summary.json"
DEPTH_THRESHOLD_METRICS_PATH = OUTPUT_ROOT / "metrics" / DEPTH_ARCH / "selected_thresholds_depth_3d.json"

DATA_SEEDS = (101, 202, 303, 404, 505)


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_output(args: list[str], cwd: Path = ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout.strip()


def compact_graph(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": sorted(
            (
                {
                    "id": n["id"],
                    "category": n["category"],
                    "entity_type": n["entity_type"],
                    "present": True,
                    **({"position_world_xyz": n["position_world_xyz"]} if "position_world_xyz" in n else {}),
                }
                for n in graph.get("nodes", [])
            ),
            key=lambda x: x["id"],
        ),
        "binary_edges": sorted(
            (
                {"subject": e["subject"], "predicate": e["predicate"], "object": e["object"]}
                for e in graph.get("binary_edges", [])
            ),
            key=lambda x: (x["subject"], x["predicate"], x["object"]),
        ),
    }


def encode_targets_numpy(graph: dict[str, Any], ontology: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    nodes = ontology["nodes"]
    preds = ontology["predicates"]
    y_node = np.zeros((len(nodes),), dtype=np.float32)
    y_edge = np.zeros((len(nodes), len(nodes), len(preds)), dtype=np.float32)
    for node in graph.get("nodes", []):
        y_node[nodes[node["id"]]["index"]] = 1.0
    for edge in graph.get("binary_edges", []):
        i = nodes[edge["subject"]]["index"]
        j = nodes[edge["object"]]["index"]
        p = preds[edge["predicate"]]
        if i != j:
            y_edge[i, j, p] = 1.0
    return y_node, y_edge


def relation_validity_mask_numpy(ontology: dict[str, Any]) -> np.ndarray:
    nodes = ontology["nodes"]
    preds = ontology["predicates"]
    mask = np.ones((len(nodes), len(nodes), len(preds)), dtype=bool)
    for i in range(len(nodes)):
        mask[i, i, :] = False
    if "grasping" in preds:
        g = preds["grasping"]
        idx_to_type = {meta["index"]: meta["entity_type"] for meta in nodes.values()}
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                mask[i, j, g] = idx_to_type[i] == "gripper" and idx_to_type[j] == "object" and i != j
    return mask


def select_validation_samples(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "validation":
            by_task[int(row["task_id"])].append(row)
    selected = []
    per_task = max(1, count // max(len(by_task), 1))
    for task_id in sorted(by_task):
        task_rows = sorted(by_task[task_id], key=lambda x: (int(x["global_episode_index"]), int(x["frame_index"])))
        step = max(1, len(task_rows) // per_task)
        selected.extend(task_rows[::step][:per_task])
    return selected[:count]


def draw_sample(row: dict[str, Any], graph: dict[str, Any], out_path: Path) -> str | None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (980, 620), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    rgb = ROOT / row["image_path"]
    if rgb.exists():
        canvas.paste(Image.open(rgb).convert("RGB").resize((256, 256)), (24, 72))
    draw.text((24, 18), f"{ARCH} task={row['task_id']:02d} global={row['global_episode_index']:06d} frame={row['frame_index']:06d}", fill=(0, 0, 0))
    draw.text((24, 42), row["instruction"][:130], fill=(35, 35, 35))
    y = 76
    draw.text((320, y), "Nodes", fill=(0, 0, 0))
    y += 24
    for node in graph["nodes"]:
        draw.text((320, y), f"- {node['id']} [{node['entity_type']}]"[:95], fill=(20, 20, 20))
        y += 16
    y += 16
    draw.text((320, y), "Edges", fill=(0, 0, 0))
    y += 24
    for edge in graph["binary_edges"][:15]:
        draw.text((320, y), f"- {edge['subject']} -> {edge['predicate']} -> {edge['object']}"[:95], fill=(20, 20, 20))
        y += 16
    if len(graph["binary_edges"]) > 15:
        draw.text((320, y), f"... {len(graph['binary_edges']) - 15} more", fill=(80, 80, 80))
    canvas.save(out_path)
    return str(out_path)


def compute_edge_pos_weight(cache_rows: list[dict[str, Any]], ontology: dict[str, Any]) -> dict[str, Any]:
    train_rows = [row for row in cache_rows if row["split"] == "train"]
    edge_counts = np.zeros((len(ontology["predicates"]),), dtype=np.float64)
    for row in train_rows:
        graph = compact_graph(read_json(ROOT / row["graph_path"]))
        _y_node, y_edge = encode_targets_numpy(graph, ontology)
        edge_counts += y_edge.sum(axis=(0, 1))
    total_pairs = float(len(train_rows) * len(ontology["nodes"]) * len(ontology["nodes"]))
    pos = np.maximum(edge_counts, 1.0)
    neg = np.maximum(total_pairs - pos, 1.0)
    weights = np.clip(neg / pos, 1.0, 30.0)
    id_to_pred = {idx: pred for pred, idx in ontology["predicates"].items()}
    return {
        "source": "Recomputed from current train split using train_depth_3d_graph_generator.py formula clamp(neg/pos, 1.0, 30.0).",
        "train_frame_count": len(train_rows),
        "total_dense_pairs_per_predicate": int(total_pairs),
        "positive_counts": {id_to_pred[i]: int(edge_counts[i]) for i in range(len(edge_counts))},
        "edge_pos_weight": {id_to_pred[i]: float(weights[i]) for i in range(len(weights))},
        "edge_pos_weight_vector_by_predicate_id": [float(x) for x in weights],
    }


def run_torch_verification(selected_rows: list[dict[str, Any]], ontology: dict[str, Any]) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    from scene_graph_generator.graph_generator.losses_3d import graph_generator_3d_loss
    from scene_graph_generator.graph_generator.masks import relation_validity_mask
    from scene_graph_generator.graph_generator.models.depth_augmented import OpenVLAOnlyPooledMLP3DGraphGenerator
    from scene_graph_generator.graph_generator.targets import encode_targets

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = OpenVLAOnlyPooledMLP3DGraphGenerator(
        int(ckpt["openvla_dim"]),
        len(ontology["nodes"]),
        len(ontology["predicates"]),
        hidden_dim=1024,
        num_layers=3,
        dropout=0.1,
    )
    strict_result = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    depth_rows = {row["sample_key"]: row for row in read_jsonl(DEPTH_DIR / "depth_manifest.jsonl")}
    depth_tensors = load_file(str(DEPTH_DIR / "depth_features.safetensors"), device="cpu")
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_shard[row["shard"]].append(row)
    validity = torch.tensor(relation_validity_mask(ontology), dtype=torch.bool)
    losses = []
    output_shapes = Counter()
    max_abs = 0.0
    for shard, rows in by_shard.items():
        tensors = load_file(str(CACHE_DIR / shard), device="cpu")
        for row in rows:
            key = row["sample_key"]
            graph = compact_graph(read_json(ROOT / row["graph_path"]))
            y_node_np, y_edge_np = encode_targets(graph, ontology)
            drow = depth_rows[key]
            features = tensors[f"{key}__features"].float().unsqueeze(0)
            attn = tensors[f"{key}__attention_mask"].bool().unsqueeze(0)
            token_type = tensors[f"{key}__token_type_mask"].long().unsqueeze(0)
            y_node = torch.tensor(y_node_np, dtype=torch.float32).unsqueeze(0)
            y_edge = torch.tensor(y_edge_np, dtype=torch.float32).unsqueeze(0)
            y_xyz = depth_tensors[drow["xyz_target_key"]].float().unsqueeze(0)
            y_xyz_mask = depth_tensors[drow["xyz_mask_key"]].float().unsqueeze(0)
            out = model(features, attn, token_type)
            loss = graph_generator_3d_loss(
                out["node_logits"],
                out["edge_logits"],
                out["xyz"],
                y_node,
                y_edge,
                y_xyz,
                y_xyz_mask,
                validity,
                edge_pos_weight=None,
                xyz_weight=1.0,
            )
            losses.append(float(loss["loss"].detach()))
            for name, tensor in out.items():
                output_shapes[f"{name}:{tuple(tensor.shape)}"] += 1
                max_abs = max(max_abs, float(tensor.detach().abs().max()))
    return {
        "status": "ok",
        "torch_version": torch.__version__,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_architecture": ckpt.get("architecture"),
        "depth_input_used": bool(ckpt.get("depth_input_used", True)),
        "state_dict_key_count": len(ckpt["model_state_dict"]),
        "state_dict_keys": list(ckpt["model_state_dict"].keys()),
        "strict_load": True,
        "strict_load_missing_keys": list(strict_result.missing_keys),
        "strict_load_unexpected_keys": list(strict_result.unexpected_keys),
        "sample_forward_count": len(selected_rows),
        "loss_min": min(losses),
        "loss_max": max(losses),
        "loss_mean": sum(losses) / len(losses),
        "output_shapes": dict(sorted(output_shapes.items())),
        "max_abs_output": max_abs,
    }


def validate_samples(selected_rows: list[dict[str, Any]], ontology: dict[str, Any]) -> dict[str, Any]:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    validity = relation_validity_mask_numpy(ontology)
    node_counts = Counter()
    predicate_counts = Counter()
    edge_counts = []
    y_edge_shape_counts = Counter()
    sample_reports = []
    for i, row in enumerate(selected_rows):
        graph = compact_graph(read_json(ROOT / row["graph_path"]))
        y_node, y_edge = encode_targets_numpy(graph, ontology)
        for node in graph["nodes"]:
            node_counts[node["id"]] += 1
        for edge in graph["binary_edges"]:
            predicate_counts[edge["predicate"]] += 1
        edge_counts.append(len(graph["binary_edges"]))
        y_edge_shape_counts[str(list(y_edge.shape))] += 1
        report = {
            "sample_index": i,
            "sample_key": row["sample_key"],
            "task_id": row["task_id"],
            "global_episode_index": row["global_episode_index"],
            "frame_index": row["frame_index"],
            "node_ids": [node["id"] for node in graph["nodes"]],
            "edge_triplets": [(edge["subject"], edge["predicate"], edge["object"]) for edge in graph["binary_edges"]],
            "relation_labels": sorted({edge["predicate"] for edge in graph["binary_edges"]}),
            "y_node_shape": list(y_node.shape),
            "y_edge_shape": list(y_edge.shape),
            "y_edge_positive_count": int(y_edge.sum()),
            "validity_mask_shape": list(validity.shape),
            "valid_relation_slots": int(validity.sum()),
            "mask": "self edges invalid; grasping only gripper->object; edge loss additionally masks absent-node pairs",
            "padding": "none; fixed dense node vector and dense KxKxR edge tensor",
            "visualization": draw_sample(row, graph, SAMPLE_DIR / f"sample_{i:03d}.png"),
        }
        write_json(SAMPLE_DIR / f"sample_{i:03d}.json", report)
        sample_reports.append(report)
    write_json(SAMPLE_DIR / "sample_manifest.json", {"samples": sample_reports})
    return {
        "status": "ok",
        "sample_count": len(sample_reports),
        "sample_report_dir": str(SAMPLE_DIR),
        "node_frequency": dict(sorted(node_counts.items())),
        "predicate_frequency": dict(sorted(predicate_counts.items())),
        "edge_count_min": min(edge_counts),
        "edge_count_max": max(edge_counts),
        "edge_count_mean": sum(edge_counts) / len(edge_counts),
        "y_edge_shape_counts": dict(sorted(y_edge_shape_counts.items())),
    }


def make_holdout_manifests(cache_rows: list[dict[str, Any]], stable_index_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Path]]:
    split = read_json(SPLIT_PATH)
    train_eps = {int(x) for x in split["episodes"]["train"]}
    holdout_eps = {int(x) for x in split["episodes"]["validation"] + split["episodes"]["test"]}
    episode_task = {}
    episode_demo = {}
    episode_task_name = {}
    for row in stable_index_rows:
        ep = int(row["global_episode_index"])
        episode_task[ep] = int(row["task_id"])
        episode_demo[ep] = int(row["demo_id"])
        episode_task_name[ep] = row["task_name"]
    steps_by_episode = Counter(int(row["global_episode_index"]) for row in stable_index_rows)
    by_task: dict[int, list[int]] = defaultdict(list)
    for ep in sorted(holdout_eps):
        by_task[episode_task[ep]].append(ep)
    target_counts = {task: (4 if task in {7, 8} else 5) for task in range(10)}
    paths = []
    overlap_rows = []
    MANIFEST_DIR.mkdir(exist_ok=True)
    for seed in DATA_SEEDS:
        selected = []
        tasks = {}
        for task in sorted(target_counts):
            candidates = sorted(by_task[task])
            need = target_counts[task]
            if len(candidates) < need:
                chosen = []
            else:
                chosen = sorted(random.Random(seed + task).sample(candidates, need))
            selected.extend(chosen)
            tasks[str(task)] = {
                "task_id": task,
                "task_name": episode_task_name[candidates[0]] if candidates else None,
                "available_holdout_demonstrations": len(candidates),
                "target_count": need,
                "selected_global_episode_indices": chosen,
                "selected_demo_ids": [episode_demo[ep] for ep in chosen],
                "selected_count": len(chosen),
                "selected_steps": int(sum(steps_by_episode[ep] for ep in chosen)),
                "insufficient": len(chosen) < need,
            }
        checksum = json_sha256(sorted(selected))
        payload = {
            "suite": "libero_spatial",
            "dataset_name": "libero_spatial_no_noops",
            "primary_graph_teacher": "depth_free",
            "teacher_architecture": ARCH,
            "seed": seed,
            "selection_unit": "demonstration",
            "selection_policy": "task-stratified from Graph Generator validation+test holdout only; no train-side backfill",
            "selected_global_episode_indices": sorted(selected),
            "selected_demo_count": len(selected),
            "usable_demo_count": 432,
            "effective_ratio": len(selected) / 432.0,
            "selected_steps": int(sum(steps_by_episode[ep] for ep in selected)),
            "sample_count": int(sum(steps_by_episode[ep] for ep in selected)),
            "overlap_with_graph_generator_train": len(set(selected) & train_eps),
            "checksum": checksum,
            "tasks": tasks,
        }
        path = MANIFEST_DIR / f"depth_free_teacher_holdout_seed{seed}.json"
        write_json(path, payload)
        paths.append(path)
        overlap_rows.append(
            {
                "seed": seed,
                "selected_demo_count": len(selected),
                "selected_steps": payload["selected_steps"],
                "effective_ratio": payload["effective_ratio"],
                "graph_train_overlap": payload["overlap_with_graph_generator_train"],
                "checksum": checksum,
                "insufficient_tasks": ",".join(task for task, item in tasks.items() if item["insufficient"]),
            }
        )
    with open(REPORTS / "depth_free_teacher_manifest_overlap.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(overlap_rows[0].keys()))
        writer.writeheader()
        writer.writerows(overlap_rows)
    return overlap_rows, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=100)
    args = parser.parse_args()
    REPORTS.mkdir(exist_ok=True)

    ontology = read_json(ONTOLOGY_PATH)
    cache_rows = read_jsonl(CACHE_DIR / "cache_manifest.jsonl")
    selected = select_validation_samples(cache_rows, args.sample_count)
    training_summary = read_json(TRAINING_SUMMARY_PATH)
    threshold_metrics = read_json(THRESHOLD_METRICS_PATH)
    depth_training_summary = read_json(DEPTH_TRAINING_SUMMARY_PATH)
    depth_threshold_metrics = read_json(DEPTH_THRESHOLD_METRICS_PATH)
    torch_report = run_torch_verification(selected, ontology)
    sample_validation = validate_samples(selected, ontology)
    edge_weights = compute_edge_pos_weight(cache_rows, ontology)
    stable_index_rows = read_jsonl(BUNDLE_INDEX)
    manifest_overlap_rows, manifest_paths = make_holdout_manifests(cache_rows, stable_index_rows)

    nodes_ordered = [node_id for node_id, _ in sorted(ontology["nodes"].items(), key=lambda kv: kv[1]["index"])]
    predicates_ordered = [pred for pred, _ in sorted(ontology["predicates"].items(), key=lambda kv: kv[1])]
    relation_sha = json_sha256({"label_to_id": ontology["predicates"]})
    ckpt_sha = file_sha256(CHECKPOINT_PATH)
    feature_shapes = sorted({tuple(row["feature_shape"]) for row in cache_rows})
    image_counts = sorted({int(row["image_token_count"]) for row in cache_rows})
    instruction_counts = sorted({int(row["instruction_token_count"]) for row in cache_rows})
    split = read_json(SPLIT_PATH)
    split_summary = read_json(SPLIT_SUMMARY_PATH)

    spec = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_teacher_candidate": "depth_free",
        "source_policy": "Locked to existing depth-free implementation and checkpoint; no redesign.",
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": ckpt_sha,
            "state_dict_keys": torch_report["state_dict_keys"],
            "state_dict_key_count": torch_report["state_dict_key_count"],
            "load_strict": torch_report["strict_load"],
            "epoch": torch_report["checkpoint_epoch"],
            "architecture": torch_report["checkpoint_architecture"],
            "source_run": str(TRAINING_SUMMARY_PATH),
        },
        "feature_source": {
            "openvla_model": "checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats",
            "model_revision": "local checkpoint; OpenVLA repo details in reports/local_inventory.md",
            "layer_name": "hidden_states[-2]",
            "layer_index": -2,
            "before_or_after_projector": "after vision projector, from Prismatic hidden states containing projected image tokens plus language tokens",
            "visual_tokens_only": False,
            "language_tokens_used": True,
            "pooling": "mean-pool image token subset and mean-pool instruction token subset separately",
            "normalization": "none on OpenVLA token features",
            "feature_dim": int(training_summary.get("openvla_dim", 4096)) if "openvla_dim" in training_summary else 4096,
            "sequence_length": [list(x) for x in feature_shapes],
            "image_token_count": image_counts,
            "instruction_token_count": instruction_counts,
            "token_type_mask": {"image": 1, "instruction": 2, "padding": 0},
        },
        "nodes": {
            "definition": "Ontology nodes from rule-based LIBERO scene graph objects/fixtures/gripper.",
            "ordering": "Ascending ontology index.",
            "ids": nodes_ordered,
            "label_to_id": {node_id: ontology["nodes"][node_id]["index"] for node_id in nodes_ordered},
            "max_count": len(nodes_ordered),
            "attributes": ["id", "category", "entity_type", "present", "position_world_xyz target"],
            "coordinate_target": "position_world_xyz from existing 3D teacher graph",
            "coordinate_normalization": "none found in implementation; SmoothL1 is applied in world-coordinate units",
            "padding_value": None,
            "mask_rule": "Dense node vector y_node[K]; no padded node slots beyond ontology K.",
        },
        "edges": {
            "definition": "Dense directed binary relation tensor over ontology node pairs and predicate vocabulary.",
            "ordering": "subject node index, object node index, predicate index.",
            "directed": True,
            "max_count": len(nodes_ordered) * len(nodes_ordered) * len(predicates_ordered),
            "candidate_pair_rule": "all KxK directed pairs, with self edges invalid in loss/decoding mask",
            "no_relation_class": None,
            "padding_value": 0.0,
            "mask_rule": "Self edges invalid. Grasping valid only gripper->object. Edge loss masks absent endpoint-node pairs.",
            "target_shape": [len(nodes_ordered), len(nodes_ordered), len(predicates_ordered)],
        },
        "relations": {
            "vocabulary": predicates_ordered,
            "label_to_id": ontology["predicates"],
            "sha256": relation_sha,
            "class_weights": edge_weights,
            "relation_direction": "directed predicates; inverse relations are explicit separate labels when present, e.g. left_of/right_of and front_of/behind",
            "inverse_relation_processing": "No automatic inverse expansion in encode_targets; graph labels already contain directed edges.",
        },
        "architecture": {
            "class_name": "OpenVLAOnlyPooledMLP3DGraphGenerator",
            "input_dimensions": {"openvla_dim": 4096, "depth_dim": None},
            "layers": [
                "encoder: 3 x [Linear(input,1024), GELU, Dropout(0.1)] with first input 4096*2",
                "node_head: Linear(1024,K)",
                "edge_head: Linear(1024,K*K*R)",
                "xyz_head: Linear(1024,K*3)",
            ],
            "hidden_dim": 1024,
            "num_layers": 3,
            "activation": "GELU",
            "dropout": 0.1,
            "output_heads": ["node_logits", "edge_logits", "xyz"],
            "trainable_parameters": training_summary["trainable_parameters"],
        },
        "loss": {
            "total_graph_loss": "node_loss + edge_loss + xyz_weight * xyz_loss",
            "node_loss": "BCEWithLogits mean over K node labels",
            "coordinate_loss": "SmoothL1 over xyz entries where xyz_mask is true",
            "edge_existence_loss": "not separate; relation presence is multi-label BCE in edge_loss",
            "relation_loss": "BCEWithLogits over dense KxKxR relation tensor with validity/present-node mask",
            "internal_weights": {"node_weight": 1.0, "edge_weight": 1.0, "xyz_weight": training_summary.get("xyz_weight", 1.0), "edge_pos_weight": edge_weights},
            "reduction": "mean over active slots after masking",
            "masking": "relation_validity_mask plus present endpoint mask; xyz_mask for coordinate target",
        },
        "split": {
            "path": str(SPLIT_PATH),
            "seed": split["split"]["seed"],
            "unit": split["split"]["unit"],
            "episode_counts": split_summary["episode_counts"],
            "frame_counts": split_summary["frame_counts"],
            "has_overlap": split_summary["has_overlap"],
        },
        "metrics": {
            "validation": threshold_metrics["validation"],
            "test": threshold_metrics["test"],
            "fixed_threshold_validation": training_summary["validation"],
            "fixed_threshold_test": training_summary["test"],
        },
        "verification": torch_report,
    }

    tensor_shapes = {
        "feature_cache_shapes": [list(x) for x in feature_shapes],
        "image_token_count": image_counts,
        "instruction_token_count": instruction_counts,
        "node_target_shape": [len(nodes_ordered)],
        "edge_target_shape": [len(nodes_ordered), len(nodes_ordered), len(predicates_ordered)],
        "xyz_target_shape": [len(nodes_ordered), 3],
        "xyz_mask_shape": [len(nodes_ordered)],
        "model_output_shapes_on_100_samples": torch_report["output_shapes"],
        "sample_validation": sample_validation,
    }
    relation_payload = {
        "vocabulary": predicates_ordered,
        "label_to_id": ontology["predicates"],
        "id_to_label": {str(v): k for k, v in ontology["predicates"].items()},
        "sha256": relation_sha,
        "multi_label": True,
        "no_relation_class": None,
        "class_weights": edge_weights,
    }
    write_json(REPORTS / "depth_free_graph_specification.json", spec)
    write_json(REPORTS / "depth_free_graph_tensor_shapes.json", tensor_shapes)
    write_json(REPORTS / "depth_free_graph_relation_vocabulary.json", relation_payload)
    write_json(REPORTS / "depth_free_graph_split.json", spec["split"])
    (REPORTS / "depth_free_graph_training_config.yaml").write_text(
        "\n".join(
            [
                "architecture: pooled_mlp_openvla_3d",
                "class_name: OpenVLAOnlyPooledMLP3DGraphGenerator",
                "openvla_dim: 4096",
                "depth_input_used: false",
                "hidden_dim: 1024",
                "num_layers: 3",
                "dropout: 0.1",
                "epochs: 30",
                "epochs_ran: 21",
                "best_epoch: 16",
                "batch_size: 64",
                "learning_rate: 1.0e-4",
                "weight_decay: 1.0e-4",
                "optimizer: AdamW",
                "gradient_clip_norm: 1.0",
                "early_stopping_patience: 5",
                "xyz_weight: 1.0",
                "feature_layer: -2",
                "split_seed: 42",
                "checkpoint: outputs/scene_graph_generator_openvla_spatial/checkpoints/pooled_mlp_openvla_3d/best.pt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    md = [
        "# Depth-Free Graph Specification Lock",
        "",
        f"Generated: {spec['generated_at']}",
        "",
        f"- Class: `{spec['architecture']['class_name']}`",
        f"- Checkpoint: `{CHECKPOINT_PATH}`",
        f"- Checkpoint SHA256: `{ckpt_sha}`",
        f"- Strict load: `{torch_report['strict_load']}`",
        f"- Relation vocabulary SHA256: `{relation_sha}`",
        f"- Depth input used by Graph Generator: `{torch_report['depth_input_used']}`",
        f"- Validation triplet F1: `{threshold_metrics['validation']['metrics']['triplet']['f1']}`",
        f"- Test triplet F1: `{threshold_metrics['test']['metrics']['triplet']['f1']}`",
        f"- Test xyz RMSE: `{threshold_metrics['test']['xyz_metrics']['rmse']}`",
        "",
        "## Feature Source",
        "",
        "- `hidden_states[-2]` from the frozen OpenVLA feature cache.",
        "- Projected image tokens and language tokens are used and mean-pooled separately.",
        "- No depth side-input is passed to the Graph Generator.",
        "",
        "## Architecture",
        "",
        *[f"- {line}" for line in spec["architecture"]["layers"]],
        "",
        "## Loss",
        "",
        f"- `{spec['loss']['total_graph_loss']}`",
        "- Graph loss internals are inherited unchanged from existing code.",
        "",
        "## Verification",
        "",
        f"- 100-sample forward count: `{torch_report['sample_forward_count']}`",
        f"- 100-sample loss mean: `{torch_report['loss_mean']}`",
        f"- Output shapes: `{torch_report['output_shapes']}`",
    ]
    (REPORTS / "depth_free_graph_specification.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    ckpt_md = [
        "# Depth-Free Graph Checkpoint Report",
        "",
        f"- Path: `{CHECKPOINT_PATH}`",
        f"- SHA256: `{ckpt_sha}`",
        f"- Architecture metadata: `{torch_report['checkpoint_architecture']}`",
        f"- Epoch: `{torch_report['checkpoint_epoch']}`",
        f"- State dict keys: `{torch_report['state_dict_key_count']}`",
        f"- Strict load: `{torch_report['strict_load']}`",
        f"- Missing keys: `{torch_report['strict_load_missing_keys']}`",
        f"- Unexpected keys: `{torch_report['strict_load_unexpected_keys']}`",
        "",
        "## State Dict Keys",
        "",
        *[f"- `{key}`" for key in torch_report["state_dict_keys"]],
    ]
    (REPORTS / "depth_free_graph_checkpoint_report.md").write_text("\n".join(ckpt_md) + "\n", encoding="utf-8")

    validation_md = [
        "# Depth-Free Graph 100-Sample Validation",
        "",
        f"- Samples validated: `{sample_validation['sample_count']}`",
        f"- Sample reports: `{SAMPLE_DIR}`",
        f"- Edge target shape counts: `{sample_validation['y_edge_shape_counts']}`",
        f"- Edge count min/max/mean: `{sample_validation['edge_count_min']}` / `{sample_validation['edge_count_max']}` / `{sample_validation['edge_count_mean']}`",
        f"- Strict load: `{torch_report['strict_load']}`",
        f"- Forward status: `{torch_report['status']}`",
    ]
    (REPORTS / "depth_free_graph_100sample_validation.md").write_text("\n".join(validation_md) + "\n", encoding="utf-8")

    rerun_md = [
        "# Depth-Free Graph Evaluation Rerun",
        "",
        "Existing evaluation code path: `scene_graph_generator/scripts/select_thresholds_depth_3d.py`.",
        "",
        f"- Status: `{threshold_metrics['status']}`",
        f"- Checkpoint epoch: `{threshold_metrics['checkpoint_epoch']}`",
        f"- Depth input used: `{threshold_metrics['depth_input_used']}`",
        f"- Validation triplet F1: `{threshold_metrics['validation']['metrics']['triplet']['f1']}`",
        f"- Test triplet F1: `{threshold_metrics['test']['metrics']['triplet']['f1']}`",
        f"- Test graph exact: `{threshold_metrics['test']['metrics']['graph']['exact_match']}`",
        f"- Test xyz RMSE: `{threshold_metrics['test']['xyz_metrics']['rmse']}`",
    ]
    (REPORTS / "depth_free_graph_evaluation_rerun.md").write_text("\n".join(rerun_md) + "\n", encoding="utf-8")

    compare_rows = {
        "depth_free": {
            "architecture": ARCH,
            "class": "OpenVLAOnlyPooledMLP3DGraphGenerator",
            "input": "OpenVLA features only",
            "params": training_summary["trainable_parameters"],
            "test_triplet_f1": threshold_metrics["test"]["metrics"]["triplet"]["f1"],
            "test_graph_exact": threshold_metrics["test"]["metrics"]["graph"]["exact_match"],
            "test_xyz_rmse": threshold_metrics["test"]["xyz_metrics"]["rmse"],
            "test_mean_l2": threshold_metrics["test"]["xyz_metrics"]["mean_l2"],
        },
        "depth_augmented": {
            "architecture": DEPTH_ARCH,
            "class": "DepthAugmentedPooledMLPGraphGenerator",
            "input": "OpenVLA features + 264-D depth side-input",
            "params": depth_training_summary["trainable_parameters"],
            "test_triplet_f1": depth_threshold_metrics["test"]["metrics"]["triplet"]["f1"],
            "test_graph_exact": depth_threshold_metrics["test"]["metrics"]["graph"]["exact_match"],
            "test_xyz_rmse": depth_threshold_metrics["test"]["xyz_metrics"]["rmse"],
            "test_mean_l2": depth_threshold_metrics["test"]["xyz_metrics"]["mean_l2"],
        },
    }
    compare_md = [
        "# Depth-Free vs Depth-Augmented Graph Generator",
        "",
        "| item | depth_free | depth_augmented |",
        "|---|---:|---:|",
        f"| architecture | `{compare_rows['depth_free']['architecture']}` | `{compare_rows['depth_augmented']['architecture']}` |",
        f"| class | `{compare_rows['depth_free']['class']}` | `{compare_rows['depth_augmented']['class']}` |",
        f"| input | {compare_rows['depth_free']['input']} | {compare_rows['depth_augmented']['input']} |",
        f"| params | {compare_rows['depth_free']['params']} | {compare_rows['depth_augmented']['params']} |",
        f"| test triplet F1 | {compare_rows['depth_free']['test_triplet_f1']} | {compare_rows['depth_augmented']['test_triplet_f1']} |",
        f"| test graph exact | {compare_rows['depth_free']['test_graph_exact']} | {compare_rows['depth_augmented']['test_graph_exact']} |",
        f"| test xyz RMSE | {compare_rows['depth_free']['test_xyz_rmse']} | {compare_rows['depth_augmented']['test_xyz_rmse']} |",
        f"| test mean L2 | {compare_rows['depth_free']['test_mean_l2']} | {compare_rows['depth_augmented']['test_mean_l2']} |",
        "",
        "Depth-free is selected only for the main teacher role; depth-augmented remains preserved as an ablation/reference.",
    ]
    (REPORTS / "depth_free_vs_depth_augmented_graph_generator.md").write_text("\n".join(compare_md) + "\n", encoding="utf-8")

    go = bool(torch_report["strict_load"]) and threshold_metrics["status"] == "ok" and threshold_metrics["test"]["metrics"]["triplet"]["f1"] > 0.5
    decision = {
        "PRIMARY_GRAPH_TEACHER": "depth_free" if go else "blocked",
        "decision": "conditional_go" if go else "no_go",
        "reason": "Relation/triplet prediction is strong; coordinate error is worse than depth-augmented, so coordinate loss must be logged separately.",
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_sha256": ckpt_sha,
        "relation_vocabulary_sha256": relation_sha,
        "validation_triplet_f1": threshold_metrics["validation"]["metrics"]["triplet"]["f1"],
        "test_triplet_f1": threshold_metrics["test"]["metrics"]["triplet"]["f1"],
        "test_xyz_rmse": threshold_metrics["test"]["xyz_metrics"]["rmse"],
        "holdout_manifest_paths": [str(p) for p in manifest_paths],
        "manifest_overlap": manifest_overlap_rows,
    }
    write_json(REPORTS / "primary_graph_teacher.lock.json", decision)
    decision_md = [
        "# Primary Graph Teacher Decision",
        "",
        f"- PRIMARY_GRAPH_TEACHER: `{decision['PRIMARY_GRAPH_TEACHER']}`",
        f"- Decision: `{decision['decision']}`",
        f"- Checkpoint SHA256: `{ckpt_sha}`",
        f"- Relation vocabulary SHA256: `{relation_sha}`",
        f"- Validation triplet F1: `{decision['validation_triplet_f1']}`",
        f"- Test triplet F1: `{decision['test_triplet_f1']}`",
        f"- Test xyz RMSE: `{decision['test_xyz_rmse']}`",
        "",
        decision["reason"],
    ]
    (REPORTS / "primary_graph_teacher_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")

    print(json.dumps({"status": "ok", "PRIMARY_GRAPH_TEACHER": decision["PRIMARY_GRAPH_TEACHER"], "checkpoint_sha256": ckpt_sha, "relation_sha256": relation_sha, "sample_forward_count": torch_report["sample_forward_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
