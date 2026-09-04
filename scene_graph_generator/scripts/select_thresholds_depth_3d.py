#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph_generator.graph_generator.decoding import sigmoid
from scene_graph_generator.graph_generator.masks import relation_validity_mask
from scene_graph_generator.graph_generator.metrics_3d import xyz_metrics
from scene_graph_generator.graph_generator.schema import compact_graph, read_json, write_json
from scene_graph_generator.graph_generator.targets import encode_targets
from scene_graph_generator.scripts.select_thresholds import best_binary_threshold, metric_key, score_arrays


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pad_batch(items, tensors, depth_tensors):
    import torch

    max_len = max(tensors[f"{x['sample_key']}__features"].shape[0] for x in items)
    dim = tensors[f"{items[0]['sample_key']}__features"].shape[1]
    depth_dim = depth_tensors[f"{items[0]['sample_key']}__depth_features"].shape[0]
    b = len(items)
    features = torch.zeros(b, max_len, dim, dtype=torch.bfloat16)
    attn = torch.zeros(b, max_len, dtype=torch.bool)
    token_type = torch.zeros(b, max_len, dtype=torch.long)
    depth = torch.zeros(b, depth_dim, dtype=torch.float32)
    for i, item in enumerate(items):
        key = item["sample_key"]
        feat = tensors[f"{key}__features"]
        n = feat.shape[0]
        features[i, :n] = feat
        attn[i, :n] = tensors[f"{key}__attention_mask"].bool()
        token_type[i, :n] = tensors[f"{key}__token_type_mask"].long()
        depth[i] = depth_tensors[f"{key}__depth_features"].float()
    return features.float(), attn, token_type, depth


def collect_split_arrays(args, split: str, model, ontology: dict, depth_rows: dict, depth_tensors, device: str) -> dict:
    import torch
    from safetensors.torch import load_file

    rows = [row for row in read_jsonl(args.openvla_cache_dir / "cache_manifest.jsonl") if row["split"] == split]
    split_by_shard = defaultdict(list)
    y_nodes = []
    y_edges = []
    y_xyz = []
    y_xyz_mask = []
    row_order = []
    for row in rows:
        if row["sample_key"] not in depth_rows:
            raise RuntimeError(f"Missing depth row for {row['sample_key']}")
        graph = compact_graph(read_json(Path(row["graph_path"])))
        y_node_np, y_edge_np = encode_targets(graph, ontology)
        drow = depth_rows[row["sample_key"]]
        item = {**row, "y_node_np": y_node_np, "y_edge_np": y_edge_np, **drow}
        split_by_shard[row["shard"]].append(item)
    node_probs = []
    edge_probs = []
    xyz_pred_all = []
    model.eval()
    with torch.no_grad():
        for shard_name in sorted(split_by_shard):
            shard_items = split_by_shard[shard_name]
            tensors = load_file(str(args.openvla_cache_dir / shard_name), device="cpu")
            for start in range(0, len(shard_items), args.batch_size):
                batch = shard_items[start : start + args.batch_size]
                features, attn, token_type, depth = pad_batch(batch, tensors, depth_tensors)
                features = features.to(device, non_blocking=True)
                attn = attn.to(device, non_blocking=True)
                token_type = token_type.to(device, non_blocking=True)
                if getattr(model, "depth_encoder", None) is None:
                    out = model(features, attn, token_type)
                else:
                    out = model(features, attn, token_type, depth.to(device, non_blocking=True))
                node_probs.append(sigmoid(out["node_logits"].detach().cpu().float().numpy()))
                edge_probs.append(sigmoid(out["edge_logits"].detach().cpu().float().numpy()))
                xyz_pred_all.append(out["xyz"].detach().cpu().float().numpy())
                for item in batch:
                    y_nodes.append(item["y_node_np"])
                    y_edges.append(item["y_edge_np"])
                    y_xyz.append(depth_tensors[item["xyz_target_key"]].float().numpy())
                    y_xyz_mask.append(depth_tensors[item["xyz_mask_key"]].float().numpy())
                    row_order.append(item["sample_key"])
    return {
        "node_probs": np.concatenate(node_probs),
        "edge_probs": np.concatenate(edge_probs),
        "y_node": np.stack(y_nodes).astype(bool),
        "y_edge": np.stack(y_edges).astype(bool),
        "xyz_pred": np.concatenate(xyz_pred_all),
        "xyz_gt": np.stack(y_xyz),
        "xyz_mask": np.stack(y_xyz_mask),
        "sample_keys": row_order,
    }


def select_predicate_thresholds(validation_arrays: dict, ontology: dict, validity_mask: np.ndarray, grid: list[float], node_threshold: float) -> tuple[dict, dict]:
    predicate_names = sorted(ontology["predicates"])
    global_candidates = []
    for threshold in grid:
        thresholds = {pred: threshold for pred in predicate_names}
        summary = score_arrays(validation_arrays, ontology, validity_mask, node_threshold=node_threshold, predicate_thresholds=thresholds)
        global_candidates.append({"threshold": threshold, "summary": summary, "key": metric_key(summary)})
    best_global = max(global_candidates, key=lambda row: (*row["key"], -abs(row["threshold"] - 0.5)))
    pred_thresholds = {pred: float(best_global["threshold"]) for pred in predicate_names}
    validation = best_global["summary"]
    predicate_selection = {
        "__global__": {
            "threshold": float(best_global["threshold"]),
            "macro_f1": validation["metrics"]["triplet"]["macro_f1"],
            "micro_f1": validation["metrics"]["triplet"]["f1"],
            "graph_exact_match": validation["metrics"]["graph"]["exact_match"],
        }
    }
    improved = True
    while improved:
        improved = False
        current_key = metric_key(validation)
        for pred_name in predicate_names:
            best_local = None
            for threshold in grid:
                candidate_thresholds = dict(pred_thresholds)
                candidate_thresholds[pred_name] = threshold
                summary = score_arrays(validation_arrays, ontology, validity_mask, node_threshold=node_threshold, predicate_thresholds=candidate_thresholds)
                key = metric_key(summary)
                if best_local is None or (*key, -abs(threshold - 0.5)) > (*best_local["key"], -abs(best_local["threshold"] - 0.5)):
                    best_local = {"threshold": threshold, "summary": summary, "key": key}
            if best_local is not None and best_local["key"] > current_key:
                pred_thresholds[pred_name] = float(best_local["threshold"])
                validation = best_local["summary"]
                current_key = best_local["key"]
                improved = True
                predicate_selection[pred_name] = {
                    "threshold": float(best_local["threshold"]),
                    "macro_f1": validation["metrics"]["triplet"]["macro_f1"],
                    "micro_f1": validation["metrics"]["triplet"]["f1"],
                    "graph_exact_match": validation["metrics"]["graph"]["exact_match"],
                }
    return pred_thresholds, {"validation": validation, "predicate_selection": predicate_selection}


def with_xyz(summary: dict, arrays: dict) -> dict:
    return {**summary, "xyz_metrics": xyz_metrics(arrays["xyz_pred"], arrays["xyz_gt"], arrays["xyz_mask"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openvla-cache-dir", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/feature_cache/all_frames"))
    ap.add_argument("--depth-dir", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/depth_features/all_frames"))
    ap.add_argument("--output-root", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial"))
    ap.add_argument("--architecture", default="pooled_mlp_depth_3d")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    args = ap.parse_args()

    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    from safetensors.torch import load_file

    from scene_graph_generator.graph_generator.models.depth_augmented import DepthAugmentedPooledMLPGraphGenerator, OpenVLAOnlyPooledMLP3DGraphGenerator

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    ontology = read_json(args.output_root / "ontology" / "ontology.json")
    validity_mask = relation_validity_mask(ontology)
    depth_rows = {row["sample_key"]: row for row in read_jsonl(args.depth_dir / "depth_manifest.jsonl")}
    depth_tensors = load_file(str(args.depth_dir / "depth_features.safetensors"), device="cpu")
    checkpoint_path = args.output_root / "checkpoints" / args.architecture / "best.pt"
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    depth_input_used = bool(ckpt.get("depth_input_used", True))
    if depth_input_used:
        model = DepthAugmentedPooledMLPGraphGenerator(
            int(ckpt["openvla_dim"]),
            int(ckpt["depth_dim"]),
            len(ontology["nodes"]),
            len(ontology["predicates"]),
            hidden_dim=1024,
            num_layers=3,
            dropout=0.1,
        ).to(args.device)
    else:
        model = OpenVLAOnlyPooledMLP3DGraphGenerator(
            int(ckpt["openvla_dim"]),
            len(ontology["nodes"]),
            len(ontology["predicates"]),
            hidden_dim=1024,
            num_layers=3,
            dropout=0.1,
        ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    validation_arrays = collect_split_arrays(args, "validation", model, ontology, depth_rows, depth_tensors, args.device)
    test_arrays = collect_split_arrays(args, "test", model, ontology, depth_rows, depth_tensors, args.device)
    grid = [float(x) for x in args.grid.split(",") if x.strip()]
    node_scores = validation_arrays["node_probs"].reshape(-1)
    node_targets = validation_arrays["y_node"].reshape(-1)
    node_selection = best_binary_threshold(node_scores, node_targets, grid)
    node_threshold = float(node_selection["threshold"])
    predicate_thresholds, selected = select_predicate_thresholds(validation_arrays, ontology, validity_mask, grid, node_threshold)
    validation = with_xyz(selected["validation"], validation_arrays)
    test = with_xyz(
        score_arrays(test_arrays, ontology, validity_mask, node_threshold=node_threshold, predicate_thresholds=predicate_thresholds),
        test_arrays,
    )
    out = {
        "status": "ok",
        "architecture": args.architecture,
        "selection_split": "validation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "depth_input_used": depth_input_used,
        "node_threshold": node_threshold,
        "node_selection": node_selection,
        "predicate_thresholds": predicate_thresholds,
        "predicate_selection": selected["predicate_selection"],
        "candidate_grid": grid,
        "validation": validation,
        "test": test,
        "elapsed_sec": round(time.time() - started, 3),
    }
    metric_dir = args.output_root / "metrics" / args.architecture
    write_json(metric_dir / "selected_thresholds_depth_3d.json", out)
    write_json(metric_dir / "validation_metrics_selected_thresholds.json", validation)
    write_json(metric_dir / "test_metrics_selected_thresholds.json", test)
    print(json.dumps({"status": "ok", "output": str(metric_dir / "selected_thresholds_depth_3d.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
