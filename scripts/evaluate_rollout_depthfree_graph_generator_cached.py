#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph_generator.graph_generator.masks import relation_validity_mask
from scene_graph_generator.graph_generator.metrics_3d import xyz_metrics
from scene_graph_generator.graph_generator.schema import read_json, write_json
from scene_graph_generator.scripts.select_thresholds import best_binary_threshold, score_arrays
from scripts.build_rollout_openvla_feature_cache import raw_instruction_from_metadata
from scripts.evaluate_rollout_depthfree_graph_generator_live import (
    add_xyz,
    load_model,
    parse_grid,
    select_predicate_thresholds,
    summarize_grasp,
)
from scripts.rollout_xyz_targets import RolloutXyzTargetCache
from scripts.train_rollout_depthfree_graph_generator import NpzCache, group_by_shard


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def collect_split_arrays(args: argparse.Namespace, split: str, model, ontology: dict[str, Any], device: str) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    rows = [row for row in read_jsonl(args.cache_dir / "cache_manifest.jsonl") if row["split"] == split]
    if args.max_frames is not None:
        rows = rows[: args.max_frames]
    if not rows:
        raise RuntimeError(f"No rows for split={split}")

    input_dim = int(rows[0]["feature_shape"][-1])
    npz_cache = NpzCache()
    xyz_sidecar_root = args.xyz_sidecar_root or (args.data_root / "inspection" / "graph3d_positions_all")
    xyz_cache = RolloutXyzTargetCache(ontology, data_root=args.data_root, sidecar_root=xyz_sidecar_root)
    node_probs = []
    edge_probs = []
    xyz_pred = []
    y_node = []
    y_edge = []
    xyz_gt = []
    xyz_mask = []
    row_order = []
    frame_meta = []
    actions = []

    grouped = group_by_shard(rows)
    total_batches = sum((len(shard_rows) + args.batch_size - 1) // args.batch_size for _shard, shard_rows in grouped)
    batch_index = 0
    started = time.time()
    for shard_name, shard_rows in grouped:
        tensors = load_file(str(args.cache_dir / shard_name), device="cpu")
        for start in range(0, len(shard_rows), args.batch_size):
            batch_index += 1
            batch = shard_rows[start : start + args.batch_size]
            max_len = max(tensors[f"{row['sample_key']}__features"].shape[0] for row in batch)
            bsz = len(batch)
            features = torch.zeros(bsz, max_len, input_dim, dtype=torch.bfloat16)
            attn = torch.zeros(bsz, max_len, dtype=torch.bool)
            token_type = torch.zeros(bsz, max_len, dtype=torch.long)
            for i, row in enumerate(batch):
                key = row["sample_key"]
                feat = tensors[f"{key}__features"]
                n = feat.shape[0]
                features[i, :n] = feat
                attn[i, :n] = tensors[f"{key}__attention_mask"].bool()
                token_type[i, :n] = tensors[f"{key}__token_type_mask"].long()

            with torch.inference_mode():
                out = model(features.float().to(device), attn.to(device), token_type.to(device))
            node_probs.append(sigmoid(out["node_logits"].detach().cpu().float().numpy()))
            edge_probs.append(sigmoid(out["edge_logits"].detach().cpu().float().numpy()))
            xyz_pred.append(out["xyz"].detach().cpu().float().numpy())

            for row in batch:
                arrays = npz_cache.get(str(row["frames_npz"]))
                idx = int(row["frame_index"])
                meta = read_json(Path(row["episode_dir"]) / "metadata.json")
                frames = meta.get("frames") or []
                xyz, mask, _xyz_source = xyz_cache.get(Path(row["episode_dir"]), arrays, idx)
                row_copy = dict(row)
                row_copy["instruction"] = raw_instruction_from_metadata(meta)
                y_node.append(arrays["node_valid_mask"][idx].astype(np.float32))
                y_edge.append(arrays["oracle_graph_tensor"][idx].astype(np.float32))
                xyz_gt.append(xyz)
                xyz_mask.append(mask)
                row_order.append(row_copy)
                frame_meta.append(frames[idx] if idx < len(frames) else {})
                actions.append(arrays["executed_action"][idx].astype(np.float32))

            if args.progress_interval and (batch_index == 1 or batch_index % args.progress_interval == 0 or batch_index == total_batches):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[{split}] batch {batch_index}/{total_batches} frames={len(row_order)}/{len(rows)} rate={len(row_order)/elapsed:.2f} frame/s",
                    flush=True,
                )

    return {
        "split": split,
        "rows": row_order,
        "frame_meta": frame_meta,
        "actions": actions,
        "node_probs": np.concatenate(node_probs),
        "edge_probs": np.concatenate(edge_probs),
        "xyz_pred": np.concatenate(xyz_pred),
        "y_node": np.stack(y_node).astype(bool),
        "y_edge": np.stack(y_edge).astype(bool),
        "xyz_gt": np.stack(xyz_gt),
        "xyz_mask": np.stack(xyz_mask),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rollout GraphGenerator from cached Base OpenVLA features.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/openvla_rollout_graph_v2/openvla_feature_cache/base_openvla_rollout_v2"))
    parser.add_argument("--data-root", type=Path, default=Path("data/openvla_rollout_graph_v2"))
    parser.add_argument("--ontology", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/ontology/ontology.json"))
    parser.add_argument("--graph-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rollout_depthfree_graph_generator_cached_eval"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--node-threshold", type=float, default=0.5)
    parser.add_argument("--predicate-threshold", type=float, default=0.5)
    parser.add_argument("--grasp-threshold", type=float, default=0.5)
    parser.add_argument("--select-thresholds", action="store_true")
    parser.add_argument("--threshold-grid", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--xyz-sidecar-root", type=Path, default=None)
    parser.add_argument("--require-xyz-targets", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--top-fp", type=int, default=25)
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")

    started = time.time()
    ontology = read_json(args.ontology)
    validity_np = relation_validity_mask(ontology)
    model, graph_ckpt = load_model(args.graph_checkpoint, ontology, args.device)
    grid = parse_grid(args.threshold_grid)

    if args.select_thresholds:
        validation_arrays = collect_split_arrays(args, "validation", model, ontology, args.device)
        eval_arrays = collect_split_arrays(args, args.split, model, ontology, args.device)
        node_selection = best_binary_threshold(validation_arrays["node_probs"].reshape(-1), validation_arrays["y_node"].reshape(-1), grid)
        node_threshold = float(node_selection["threshold"])
        predicate_thresholds, selected = select_predicate_thresholds(validation_arrays, ontology, validity_np, grid, node_threshold)
        validation_summary = add_xyz(selected["validation"], validation_arrays)
        eval_summary = add_xyz(
            score_arrays(eval_arrays, ontology, validity_np, node_threshold=node_threshold, predicate_thresholds=predicate_thresholds),
            eval_arrays,
        )
        eval_grasp = summarize_grasp(
            eval_arrays,
            ontology,
            validity_np,
            node_threshold=node_threshold,
            predicate_thresholds=predicate_thresholds,
            output_root=args.output_root,
            save_predictions=args.save_predictions,
            top_fp=args.top_fp,
        )
        report = {
            "status": "ok",
            "cache_dir": str(args.cache_dir),
            "graph_checkpoint": str(args.graph_checkpoint),
            "graph_checkpoint_epoch": graph_ckpt.get("epoch"),
            "threshold_selection": {
                "candidate_grid": grid,
                "node_threshold": node_threshold,
                "node_selection": node_selection,
                "predicate_thresholds": predicate_thresholds,
                "predicate_selection": selected["predicate_selection"],
            },
            "validation": validation_summary,
            args.split: {**eval_summary, **eval_grasp},
            "elapsed_sec": round(time.time() - started, 3),
        }
        out_path = args.output_root / "metrics" / f"{args.split}_selected_thresholds_grasp_eval.json"
    else:
        arrays = collect_split_arrays(args, args.split, model, ontology, args.device)
        thresholds = {name: args.predicate_threshold for name in ontology["predicates"]}
        thresholds["grasping"] = args.grasp_threshold
        eval_summary = add_xyz(score_arrays(arrays, ontology, validity_np, node_threshold=args.node_threshold, predicate_thresholds=thresholds), arrays)
        eval_grasp = summarize_grasp(
            arrays,
            ontology,
            validity_np,
            node_threshold=args.node_threshold,
            predicate_thresholds=thresholds,
            output_root=args.output_root,
            save_predictions=args.save_predictions,
            top_fp=args.top_fp,
        )
        report = {
            "status": "ok",
            "cache_dir": str(args.cache_dir),
            "split": args.split,
            "graph_checkpoint": str(args.graph_checkpoint),
            "graph_checkpoint_epoch": graph_ckpt.get("epoch"),
            "thresholds": {"node": args.node_threshold, "predicate_default": args.predicate_threshold, "grasping": args.grasp_threshold},
            args.split: {**eval_summary, **eval_grasp},
            "elapsed_sec": round(time.time() - started, 3),
        }
        out_path = args.output_root / "metrics" / f"{args.split}_grasp_eval.json"

    xyz = report[args.split]["xyz_metrics"]
    if args.require_xyz_targets and int(xyz.get("num_points", 0)) <= 0:
        raise RuntimeError("No valid XYZ targets found for cached evaluation.")
    write_json(out_path, report)
    print(json.dumps({"status": "ok", "output": str(out_path), "elapsed_sec": report["elapsed_sec"]}, sort_keys=True))


if __name__ == "__main__":
    main()
