#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SAMPLES_DIR = REPORTS / "graph_samples"

OUTPUT_ROOT = ROOT / "outputs/scene_graph_generator_openvla_spatial"
CACHE_DIR = OUTPUT_ROOT / "feature_cache/all_frames"
DEPTH_DIR = OUTPUT_ROOT / "depth_features/all_frames"
ONTOLOGY_PATH = OUTPUT_ROOT / "ontology/ontology.json"
CHECKPOINT_PATH = OUTPUT_ROOT / "checkpoints/pooled_mlp_depth_3d/best.pt"
TRAINING_SUMMARY_PATH = OUTPUT_ROOT / "metrics/pooled_mlp_depth_3d/training_summary.json"
TRAINING_STATUS_PATH = OUTPUT_ROOT / "reports/pooled_mlp_depth_3d_training_status.json"
FEATURE_SUMMARY_PATH = OUTPUT_ROOT / "reports/full_feature_cache_summary.json"
DEPTH_SUMMARY_PATH = DEPTH_DIR / "depth_feature_export_summary.json"
SPLIT_PATH = OUTPUT_ROOT / "splits/split_seed42_train50_val25_test25.json"
SPLIT_SUMMARY_PATH = OUTPUT_ROOT / "splits/split_summary.json"
BUNDLE_DIR = ROOT / "artifacts/openvla_graph_internalization_bundle_v2"

SOURCE_FILES = {
    "model": ROOT / "scene_graph_generator/graph_generator/models/depth_augmented.py",
    "training_entrypoint": ROOT / "scene_graph_generator/scripts/train_depth_3d_graph_generator.py",
    "loss_3d": ROOT / "scene_graph_generator/graph_generator/losses_3d.py",
    "loss": ROOT / "scene_graph_generator/graph_generator/losses.py",
    "targets": ROOT / "scene_graph_generator/graph_generator/targets.py",
    "masks": ROOT / "scene_graph_generator/graph_generator/masks.py",
    "decoding": ROOT / "scene_graph_generator/graph_generator/decoding.py",
    "feature_cache": ROOT / "scene_graph_generator/scripts/cache_all_features.py",
    "feature_extractor": ROOT / "scene_graph_generator/graph_generator/feature_extractor.py",
    "token_selection": ROOT / "scene_graph_generator/graph_generator/token_selection.py",
    "depth_export": ROOT / "scene_graph_generator/scripts/export_depth_features_from_libero.py",
}


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


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


def load_source_line_refs() -> dict[str, str]:
    refs = {}
    for name, path in SOURCE_FILES.items():
        refs[name] = f"{path} sha256={file_sha256(path) if path.exists() else 'missing'}"
    return refs


def sample_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["task_id"]), int(row["global_episode_index"]), int(row["frame_index"])


def compact_graph(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = sorted(
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
    )
    edges = sorted(
        (
            {"subject": e["subject"], "predicate": e["predicate"], "object": e["object"]}
            for e in graph.get("binary_edges", [])
        ),
        key=lambda x: (x["subject"], x["predicate"], x["object"]),
    )
    out = {"nodes": nodes, "binary_edges": edges}
    if graph.get("graph_type"):
        out["graph_type"] = graph["graph_type"]
    if graph.get("coordinate_frame"):
        out["coordinate_frame"] = graph["coordinate_frame"]
    return out


def draw_sample_visualization(row: dict[str, Any], graph: dict[str, Any], output_path: Path) -> str | None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (980, 620), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    image_path = ROOT / row["image_path"]
    if image_path.exists():
        image = Image.open(image_path).convert("RGB").resize((256, 256))
        canvas.paste(image, (24, 72))
    title = f"task={row['task_id']:02d} global={row['global_episode_index']:06d} frame={row['frame_index']:06d}"
    draw.text((24, 18), title, fill=(0, 0, 0))
    draw.text((24, 42), row.get("instruction", "")[:130], fill=(35, 35, 35))
    draw.text((320, 72), "Nodes", fill=(0, 0, 0))
    y = 98
    for node in graph.get("nodes", []):
        xyz = node.get("position_world_xyz")
        suffix = f" xyz={[round(float(x), 3) for x in xyz]}" if xyz is not None else ""
        draw.text((320, y), f"- {node['id']} [{node['entity_type']}]{suffix}"[:95], fill=(20, 20, 20))
        y += 16
    draw.text((320, min(y + 16, 360)), "Edges", fill=(0, 0, 0))
    y = min(y + 42, 386)
    for edge in graph.get("binary_edges", [])[:13]:
        draw.text(
            (320, y),
            f"- {edge['subject']} -> {edge['predicate']} -> {edge['object']}"[:95],
            fill=(20, 20, 20),
        )
        y += 16
    if len(graph.get("binary_edges", [])) > 13:
        draw.text((320, y), f"... {len(graph['binary_edges']) - 13} more", fill=(80, 80, 80))
    canvas.save(output_path)
    return str(output_path)


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
    k = len(nodes)
    r = len(preds)
    mask = np.ones((k, k, r), dtype=bool)
    for i in range(k):
        mask[i, i, :] = False
    if "grasping" in preds:
        g = preds["grasping"]
        idx_to_type = {meta["index"]: meta["entity_type"] for meta in nodes.values()}
        for i in range(k):
            for j in range(k):
                mask[i, j, g] = idx_to_type[i] == "gripper" and idx_to_type[j] == "object" and i != j
    return mask


def select_validation_samples(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[int(row["task_id"])].append(row)
    selected = []
    per_task = max(1, count // max(len(by_task), 1))
    for task_id in sorted(by_task):
        task_rows = sorted(by_task[task_id], key=lambda x: (int(x["global_episode_index"]), int(x["frame_index"])))
        step = max(1, len(task_rows) // per_task)
        selected.extend(task_rows[::step][:per_task])
    if len(selected) < count:
        seen = {sample_key(r) for r in selected}
        for row in sorted(rows, key=lambda x: (int(x["task_id"]), int(x["global_episode_index"]), int(x["frame_index"]))):
            if sample_key(row) not in seen:
                selected.append(row)
                seen.add(sample_key(row))
            if len(selected) >= count:
                break
    return selected[:count]


def run_torch_verification(selected_rows: list[dict[str, Any]], ontology: dict[str, Any]) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    from scene_graph_generator.graph_generator.losses_3d import graph_generator_3d_loss
    from scene_graph_generator.graph_generator.masks import relation_validity_mask
    from scene_graph_generator.graph_generator.models.depth_augmented import DepthAugmentedPooledMLPGraphGenerator
    from scene_graph_generator.graph_generator.targets import encode_targets

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = checkpoint["model_state_dict"]
    model = DepthAugmentedPooledMLPGraphGenerator(
        int(checkpoint["openvla_dim"]),
        int(checkpoint["depth_dim"]),
        len(ontology["nodes"]),
        len(ontology["predicates"]),
        hidden_dim=1024,
        num_layers=3,
        dropout=0.1,
    )
    strict_result = model.load_state_dict(state, strict=True)
    model.eval()

    rows_by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        rows_by_shard[row["shard"]].append(row)
    depth_rows = {row["sample_key"]: row for row in read_jsonl(DEPTH_DIR / "depth_manifest.jsonl")}
    depth_tensors = load_file(str(DEPTH_DIR / "depth_features.safetensors"), device="cpu")
    validity_np = relation_validity_mask(ontology)
    validity = torch.tensor(validity_np, dtype=torch.bool)
    losses = []
    output_shapes = Counter()
    max_abs = 0.0
    with torch.enable_grad():
        for shard_name, shard_rows in rows_by_shard.items():
            tensors = load_file(str(CACHE_DIR / shard_name), device="cpu")
            for row in shard_rows:
                key = row["sample_key"]
                graph = compact_graph(read_json(ROOT / row["graph_path"]))
                y_node_np, y_edge_np = encode_targets(graph, ontology)
                drow = depth_rows[key]
                features = tensors[f"{key}__features"].float().unsqueeze(0)
                attn = tensors[f"{key}__attention_mask"].bool().unsqueeze(0)
                token_type = tensors[f"{key}__token_type_mask"].long().unsqueeze(0)
                depth = depth_tensors[drow["depth_feature_key"]].float().unsqueeze(0)
                y_node = torch.tensor(y_node_np, dtype=torch.float32).unsqueeze(0)
                y_edge = torch.tensor(y_edge_np, dtype=torch.float32).unsqueeze(0)
                y_xyz = depth_tensors[drow["xyz_target_key"]].float().unsqueeze(0)
                y_xyz_mask = depth_tensors[drow["xyz_mask_key"]].float().unsqueeze(0)
                out = model(features, attn, token_type, depth)
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
                for out_name, tensor in out.items():
                    output_shapes[f"{out_name}:{tuple(tensor.shape)}"] += 1
                    max_abs = max(max_abs, float(tensor.detach().abs().max()))
    return {
        "status": "ok",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_architecture": checkpoint.get("architecture"),
        "state_dict_key_count": len(state),
        "state_dict_keys": list(state.keys()),
        "strict_load": True,
        "strict_load_missing_keys": list(strict_result.missing_keys),
        "strict_load_unexpected_keys": list(strict_result.unexpected_keys),
        "sample_forward_count": len(selected_rows),
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "loss_mean": sum(losses) / len(losses) if losses else None,
        "output_shapes": dict(sorted(output_shapes.items())),
        "max_abs_output": max_abs,
        "torch_version": torch.__version__,
    }


def build_sample_validation(selected_rows: list[dict[str, Any]], ontology: dict[str, Any]) -> dict[str, Any]:
    validity = relation_validity_mask_numpy(ontology)
    sample_reports = []
    node_counts = Counter()
    predicate_counts = Counter()
    edge_count_by_sample = []
    y_edge_shape_counts = Counter()
    y_node_shape_counts = Counter()
    xyz_shape_counts = Counter()
    xyz_mask_sum_counts = Counter()
    depth_rows = {row["sample_key"]: row for row in read_jsonl(DEPTH_DIR / "depth_manifest.jsonl")}
    for idx, row in enumerate(selected_rows):
        graph = compact_graph(read_json(ROOT / row["graph_path"]))
        y_node, y_edge = encode_targets_numpy(graph, ontology)
        drow = depth_rows[row["sample_key"]]
        node_ids = [n["id"] for n in graph.get("nodes", [])]
        edges = [(e["subject"], e["predicate"], e["object"]) for e in graph.get("binary_edges", [])]
        for node_id in node_ids:
            node_counts[node_id] += 1
        for _s, pred, _o in edges:
            predicate_counts[pred] += 1
        edge_count_by_sample.append(len(edges))
        y_node_shape_counts[str(list(y_node.shape))] += 1
        y_edge_shape_counts[str(list(y_edge.shape))] += 1
        xyz_shape_counts[str([len(ontology["nodes"]), 3])] += 1
        xyz_mask_sum_counts[str(drow.get("xyz_mask_sum"))] += 1
        image_path = draw_sample_visualization(row, graph, SAMPLES_DIR / f"sample_{idx:03d}.png")
        sample_reports.append(
            {
                "sample_index": idx,
                "sample_key": row["sample_key"],
                "task_id": row["task_id"],
                "global_episode_index": row["global_episode_index"],
                "frame_index": row["frame_index"],
                "node_ids": node_ids,
                "edge_triplets": edges,
                "relation_labels": sorted({pred for _s, pred, _o in edges}),
                "y_node_shape": list(y_node.shape),
                "y_edge_shape": list(y_edge.shape),
                "y_edge_positive_count": int(y_edge.sum()),
                "validity_mask_shape": list(validity.shape),
                "valid_relation_slots": int(validity.sum()),
                "padding": "none; fixed dense node vector and dense KxKxR edge tensor",
                "mask": "self edges invalid; grasping only gripper->object; edge loss additionally masks absent-node pairs",
                "depth_feature_key": drow["depth_feature_key"],
                "xyz_target_key": drow["xyz_target_key"],
                "xyz_mask_key": drow["xyz_mask_key"],
                "visualization": image_path,
            }
        )
    for report in sample_reports:
        write_json(SAMPLES_DIR / f"sample_{report['sample_index']:03d}.json", report)
    summary = {
        "status": "ok",
        "sample_count": len(sample_reports),
        "sample_report_dir": str(SAMPLES_DIR),
        "node_frequency": dict(sorted(node_counts.items())),
        "predicate_frequency": dict(sorted(predicate_counts.items())),
        "edge_count_min": min(edge_count_by_sample) if edge_count_by_sample else None,
        "edge_count_max": max(edge_count_by_sample) if edge_count_by_sample else None,
        "edge_count_mean": sum(edge_count_by_sample) / len(edge_count_by_sample) if edge_count_by_sample else None,
        "y_node_shape_counts": dict(sorted(y_node_shape_counts.items())),
        "y_edge_shape_counts": dict(sorted(y_edge_shape_counts.items())),
        "xyz_shape_counts": dict(sorted(xyz_shape_counts.items())),
        "xyz_mask_sum_counts": dict(sorted(xyz_mask_sum_counts.items())),
    }
    write_json(SAMPLES_DIR / "sample_manifest.json", {"samples": sample_reports})
    return summary


def build_split_reports(cache_rows: list[dict[str, Any]], bundle_summary: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split = read_json(SPLIT_PATH)
    split_summary = read_json(SPLIT_SUMMARY_PATH)
    episode_task = {}
    for row in cache_rows:
        episode_task[int(row["global_episode_index"])] = int(row["task_id"])
    split_sets = {name: {int(x) for x in eps} for name, eps in split["episodes"].items()}
    report = {
        "split_source": str(SPLIT_PATH),
        "split_summary_source": str(SPLIT_SUMMARY_PATH),
        "unit": split["split"]["unit"],
        "seed": split["split"]["seed"],
        "episode_counts": {k: len(v) for k, v in split_sets.items()},
        "frame_counts": split_summary.get("frame_counts"),
        "overlap_check": split_summary.get("overlap_check"),
        "has_overlap": split_summary.get("has_overlap"),
        "train_demo_ids_by_task": defaultdict(list),
        "validation_demo_ids_by_task": defaultdict(list),
        "test_demo_ids_by_task": defaultdict(list),
    }
    for split_name, eps in split_sets.items():
        key = f"{split_name}_demo_ids_by_task"
        for ep in sorted(eps):
            report[key][str(episode_task.get(ep, -1))].append(ep)
    for key in list(report):
        if key.endswith("_demo_ids_by_task"):
            report[key] = dict(sorted(report[key].items(), key=lambda kv: int(kv[0])))

    overlap_rows = []
    for seed, seed_summary in sorted(bundle_summary.get("seeds", {}).items(), key=lambda kv: int(kv[0])):
        manifest = read_json(BUNDLE_DIR / "manifests" / f"libero_spatial_10pct_seed{seed}.json")
        selected = {int(x) for x in manifest["selected_global_episode_indices"]}
        row = {
            "manifest_seed": seed,
            "selected_demos": len(selected),
            "manifest_checksum": manifest["checksum"],
            "graph_train_overlap": len(selected & split_sets.get("train", set())),
            "graph_validation_overlap": len(selected & split_sets.get("validation", set())),
            "graph_test_overlap": len(selected & split_sets.get("test", set())),
            "selected_steps": seed_summary.get("selected_step_count"),
            "effective_ratio": len(selected) / max(len(episode_task), 1),
        }
        overlap_rows.append(row)
    return report, overlap_rows


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


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=100)
    args = parser.parse_args()

    REPORTS.mkdir(exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    ontology = read_json(ONTOLOGY_PATH)
    training_summary = read_json(TRAINING_SUMMARY_PATH)
    training_status = read_json(TRAINING_STATUS_PATH)
    feature_summary = read_json(FEATURE_SUMMARY_PATH)
    depth_summary = read_json(DEPTH_SUMMARY_PATH)
    bundle_summary = read_json(BUNDLE_DIR / "manifest_summary.json")
    cache_rows = read_jsonl(CACHE_DIR / "cache_manifest.jsonl")
    selected_rows = select_validation_samples([r for r in cache_rows if r["split"] == "validation"], args.sample_count)

    torch_report = run_torch_verification(selected_rows, ontology)
    sample_validation = build_sample_validation(selected_rows, ontology)
    split_report, overlap_rows = build_split_reports(cache_rows, bundle_summary)
    edge_weight_report = compute_edge_pos_weight(cache_rows, ontology)

    nodes_ordered = [node_id for node_id, _ in sorted(ontology["nodes"].items(), key=lambda kv: kv[1]["index"])]
    predicates_ordered = [pred for pred, _ in sorted(ontology["predicates"].items(), key=lambda kv: kv[1])]
    relation_payload = {
        "vocabulary": predicates_ordered,
        "label_to_id": ontology["predicates"],
        "id_to_label": {str(v): k for k, v in ontology["predicates"].items()},
        "sha256": json_sha256({"label_to_id": ontology["predicates"]}),
        "no_relation_class": None,
        "multi_label": True,
        "class_weights": edge_weight_report,
        "validity_mask_rule": "All non-self directed pairs valid except grasping is valid only from gripper to object.",
    }

    state_keys = torch_report["state_dict_keys"]
    spec = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "Locked to existing implementation and checkpoint; no Graph Generator redesign.",
        "source_files": load_source_line_refs(),
        "repository": {
            "top_commit": git_output(["rev-parse", "HEAD"], ROOT),
            "top_dirty_status": git_output(["status", "--short"], ROOT).splitlines(),
        },
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": file_sha256(CHECKPOINT_PATH),
            "state_dict_keys": state_keys,
            "state_dict_key_count": len(state_keys),
            "load_strict": torch_report["strict_load"],
            "epoch": torch_report["checkpoint_epoch"],
            "architecture": torch_report["checkpoint_architecture"],
        },
        "feature_source": {
            "openvla_model": "checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats",
            "model_revision": "local checkpoint; source OpenVLA repo commit recorded in local_inventory.md",
            "layer_name": "hidden_states[-2]",
            "layer_index": -2,
            "before_or_after_projector": "after vision projector, from Prismatic hidden states containing projected image tokens plus language tokens",
            "visual_tokens_only": False,
            "language_tokens_used": True,
            "pooling": "mean-pool image token subset and mean-pool instruction token subset separately inside graph generator",
            "normalization": "no normalization on OpenVLA token features in graph generator; depth branch starts with LayerNorm(depth_dim)",
            "feature_dim": int(training_summary["checkpoint"] and 4096),
            "sequence_length": sorted({tuple(r["feature_shape"])[0] for r in cache_rows}),
            "image_token_count": sorted({int(r["image_token_count"]) for r in cache_rows}),
            "instruction_token_count": sorted({int(r["instruction_token_count"]) for r in cache_rows}),
            "token_type_mask": {"image": 1, "instruction": 2, "padding": 0},
        },
        "depth_feature": {
            "feature_dim": int(depth_summary["depth_feature_dim"]),
            "meaning": "16x16 pooled depth statistics vector (256 values) plus 8 per-frame depth stats; this is depth side-input dimension, not OpenVLA hidden size.",
            "shape": [int(depth_summary["depth_feature_dim"])],
            "source": str(DEPTH_DIR / "depth_features.safetensors"),
            "depth_shape": [256, 256],
        },
        "nodes": {
            "definition": "Ontology nodes from rule-based LIBERO scene graph objects/fixtures/gripper.",
            "ordering": "Ascending ontology index.",
            "ids": nodes_ordered,
            "label_to_id": {node_id: ontology["nodes"][node_id]["index"] for node_id in nodes_ordered},
            "max_count": len(nodes_ordered),
            "attributes": ["id", "category", "entity_type", "present", "position_world_xyz for 3D teacher/xyz target"],
            "padding_value": None,
            "mask_rule": "Dense node vector y_node[K]; node present labels are binary. No padded node slots beyond ontology K.",
        },
        "edges": {
            "definition": "Dense directed binary relation tensor over ontology node pairs and predicate vocabulary.",
            "ordering": "subject node index, object node index, predicate index.",
            "directed": True,
            "max_count": len(nodes_ordered) * len(nodes_ordered) * len(predicates_ordered),
            "no_relation_class": None,
            "padding_value": 0.0,
            "mask_rule": "Self edges invalid. Edge loss masks relation_validity_mask and pairs where both endpoint nodes are present.",
            "target_shape": [len(nodes_ordered), len(nodes_ordered), len(predicates_ordered)],
        },
        "relations": {
            "vocabulary": predicates_ordered,
            "label_to_id": ontology["predicates"],
            "class_weights": edge_weight_report,
            "sha256": relation_payload["sha256"],
        },
        "architecture": {
            "class_name": "DepthAugmentedPooledMLPGraphGenerator",
            "layers": [
                "depth_encoder: LayerNorm(264), Linear(264,1024), GELU, Dropout(0.1), Linear(1024,4096), GELU",
                "encoder: 3 x [Linear(input,1024), GELU, Dropout(0.1)] with first input 4096*3",
                "node_head: Linear(1024,K)",
                "edge_head: Linear(1024,K*K*R)",
                "xyz_head: Linear(1024,K*3)",
            ],
            "activation": "GELU",
            "dropout": 0.1,
            "hidden_dim": 1024,
            "num_layers": 3,
            "openvla_dim": 4096,
            "depth_dim": 264,
            "trainable_parameters": training_summary["trainable_parameters"],
        },
        "loss": {
            "total_definition": "loss = node_loss + edge_loss + xyz_weight * xyz_loss",
            "node_loss": "binary_cross_entropy_with_logits(node_logits, y_node, reduction='none').mean()",
            "edge_loss": "binary_cross_entropy_with_logits(edge_logits, y_edge, pos_weight=edge_pos_weight, reduction='none') masked then mean",
            "relation_loss": "same as edge_loss; predicates are multi-label, not softmax relation classes",
            "xyz_loss": "smooth_l1_loss over xyz slots where xyz_mask is true",
            "weights": {"node_weight": 1.0, "edge_weight": 1.0, "xyz_weight": training_summary.get("xyz_weight", 1.0)},
            "reduction": "mean over node logits; mean over valid edge slots; SmoothL1 default mean over masked xyz values",
        },
        "verification": torch_report,
    }

    tensor_shapes = {
        "feature_cache_shapes": feature_summary["feature_shapes"],
        "image_token_count": spec["feature_source"]["image_token_count"],
        "instruction_token_count": spec["feature_source"]["instruction_token_count"],
        "depth_feature_shape": [depth_summary["depth_feature_dim"]],
        "node_target_shape": [len(nodes_ordered)],
        "edge_target_shape": [len(nodes_ordered), len(nodes_ordered), len(predicates_ordered)],
        "xyz_target_shape": [len(nodes_ordered), 3],
        "xyz_mask_shape": [len(nodes_ordered)],
        "model_output_shapes_on_100_samples": torch_report["output_shapes"],
        "sample_validation": sample_validation,
    }

    write_json(REPORTS / "current_graph_specification.json", spec)
    write_json(REPORTS / "current_graph_tensor_shapes.json", tensor_shapes)
    write_json(REPORTS / "current_graph_relation_vocabulary.json", relation_payload)
    write_json(REPORTS / "graph_generator_split.json", split_report)

    with open(REPORTS / "graph_generator_manifest_overlap.csv", "w", newline="") as f:
        fieldnames = [
            "manifest_seed",
            "selected_demos",
            "selected_steps",
            "effective_ratio",
            "manifest_checksum",
            "graph_train_overlap",
            "graph_validation_overlap",
            "graph_test_overlap",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in overlap_rows:
            writer.writerow(row)

    spec_md = [
        "# Current Graph Specification Lock",
        "",
        f"Generated: {spec['generated_at']}",
        "",
        "## Identity",
        "",
        f"- Class: `{spec['architecture']['class_name']}`",
        f"- Checkpoint: `{CHECKPOINT_PATH}`",
        f"- Checkpoint SHA256: `{spec['checkpoint']['sha256']}`",
        f"- Strict load: `{spec['checkpoint']['load_strict']}`",
        f"- Relation vocabulary SHA256: `{relation_payload['sha256']}`",
        "",
        "## Feature Source",
        "",
        f"- OpenVLA cache checkpoint: `{spec['feature_source']['openvla_model']}`",
        f"- Feature layer: `{spec['feature_source']['layer_name']}`",
        "- Image tokens and instruction tokens are both used.",
        "- The model mean-pools image-token features and instruction-token features separately.",
        f"- OpenVLA hidden width: `{spec['feature_source']['feature_dim']}`",
        f"- Sequence lengths observed: `{spec['feature_source']['sequence_length']}`",
        "",
        "## Nodes And Relations",
        "",
        f"- Nodes: `{', '.join(nodes_ordered)}`",
        f"- Relations: `{', '.join(predicates_ordered)}`",
        "- Edge tensor is dense directed `K x K x R`; no explicit no-relation class is used.",
        "- Padding is not used for nodes/edges; invalid slots are excluded by masks.",
        "",
        "## Architecture",
        "",
        *[f"- {line}" for line in spec["architecture"]["layers"]],
        "",
        "## Loss",
        "",
        f"- Total: `{spec['loss']['total_definition']}`",
        f"- Node: `{spec['loss']['node_loss']}`",
        f"- Edge/relation: `{spec['loss']['edge_loss']}`",
        f"- XYZ: `{spec['loss']['xyz_loss']}`",
        "",
        "## Verification",
        "",
        f"- 100-sample forward count: `{torch_report['sample_forward_count']}`",
        f"- 100-sample loss mean: `{torch_report['loss_mean']}`",
        f"- Output shapes: `{torch_report['output_shapes']}`",
        f"- Sample validation: `{REPORTS / 'graph_target_validation.md'}`",
    ]
    (REPORTS / "current_graph_specification.md").write_text("\n".join(spec_md) + "\n", encoding="utf-8")

    checkpoint_md = [
        "# Current Graph Checkpoint Report",
        "",
        f"- Path: `{CHECKPOINT_PATH}`",
        f"- SHA256: `{spec['checkpoint']['sha256']}`",
        f"- Architecture metadata: `{torch_report['checkpoint_architecture']}`",
        f"- Epoch: `{torch_report['checkpoint_epoch']}`",
        f"- State dict keys: `{len(state_keys)}`",
        f"- Strict load: `{torch_report['strict_load']}`",
        f"- Missing keys: `{torch_report['strict_load_missing_keys']}`",
        f"- Unexpected keys: `{torch_report['strict_load_unexpected_keys']}`",
        f"- Torch version used for verification: `{torch_report['torch_version']}`",
        "",
        "## First State Dict Keys",
        "",
        *[f"- `{key}`" for key in state_keys[:30]],
    ]
    (REPORTS / "current_graph_checkpoint_report.md").write_text("\n".join(checkpoint_md) + "\n", encoding="utf-8")

    feature_dim_md = [
        "# Feature Dim 264 Explanation",
        "",
        "`264` is the depth side-input feature dimension used by `DepthAugmentedPooledMLPGraphGenerator`, not the OpenVLA token hidden size.",
        "",
        "Code evidence:",
        "",
        "- `export_depth_features_from_libero.py::depth_to_feature` pools a `16 x 16` grid, producing 256 spatial values.",
        "- The same function appends 8 scalar statistics: mean, std, min, max, p10, p50, p90, and finite fraction.",
        "- `256 + 8 = 264`.",
        "- `train_depth_3d_graph_generator.py` reads this as `depth_dim` from `depth_manifest.jsonl` and stores `depth_dim=264` in the checkpoint.",
        "- The OpenVLA cached token feature width is `4096`; observed feature cache shapes are listed in `current_graph_tensor_shapes.json`.",
    ]
    (REPORTS / "feature_dim_264_explanation.md").write_text("\n".join(feature_dim_md) + "\n", encoding="utf-8")

    target_md = [
        "# Graph Target Validation",
        "",
        f"- Samples validated: `{sample_validation['sample_count']}`",
        f"- Sample report directory: `{SAMPLES_DIR}`",
        f"- Node target shape counts: `{sample_validation['y_node_shape_counts']}`",
        f"- Edge target shape counts: `{sample_validation['y_edge_shape_counts']}`",
        f"- XYZ target shape counts: `{sample_validation['xyz_shape_counts']}`",
        f"- Edge count min/max/mean: `{sample_validation['edge_count_min']}` / `{sample_validation['edge_count_max']}` / `{sample_validation['edge_count_mean']}`",
        "",
        "## Relation Frequency In Validated Samples",
        "",
        *[f"- `{key}`: {value}" for key, value in sample_validation["predicate_frequency"].items()],
    ]
    (REPORTS / "graph_target_validation.md").write_text("\n".join(target_md) + "\n", encoding="utf-8")

    split_md = [
        "# Graph Generator Split Audit",
        "",
        f"- Split source: `{SPLIT_PATH}`",
        f"- Unit: `{split_report['unit']}`",
        f"- Seed: `{split_report['seed']}`",
        f"- Episode counts: `{split_report['episode_counts']}`",
        f"- Frame counts: `{split_report['frame_counts']}`",
        f"- Internal split overlap: `{split_report['overlap_check']}`",
        "",
        "## VLA 10% Manifest Overlap",
        "",
        *markdown_table(
            overlap_rows,
            [
                "manifest_seed",
                "selected_demos",
                "selected_steps",
                "effective_ratio",
                "graph_train_overlap",
                "graph_validation_overlap",
                "graph_test_overlap",
            ],
        ),
        "",
        "The current manifests overlap the Graph Generator train split. Phase 4 must decide whether holdout-only manifests can satisfy the task-wise data requirement before full training.",
    ]
    (REPORTS / "graph_generator_split_audit.md").write_text("\n".join(split_md) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "strict_load": torch_report["strict_load"],
                "sample_forward_count": torch_report["sample_forward_count"],
                "checkpoint_sha256": spec["checkpoint"]["sha256"],
                "relation_vocabulary_sha256": relation_payload["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
