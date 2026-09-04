#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph_generator.graph_generator.decoding import decode_graph
from scene_graph_generator.graph_generator.feature_extractor import OpenVLAFeatureExtractor
from scene_graph_generator.graph_generator.masks import relation_validity_mask
from scene_graph_generator.graph_generator.metrics import prf, summarize_examples
from scene_graph_generator.graph_generator.metrics_3d import xyz_metrics
from scene_graph_generator.graph_generator.schema import graph_node_ids, graph_triplets, validate_graph, write_json
from scene_graph_generator.scripts.select_thresholds import best_binary_threshold, metric_key, score_arrays
from scripts.rollout_xyz_targets import RolloutXyzTargetCache
from scripts.train_rollout_depthfree_graph_generator_live import (
    NpzCache,
    build_frame_rows,
    graph_from_targets,
    resolve_checkpoint,
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def graph_with_xyz(pred_graph: dict[str, Any], pred_xyz: np.ndarray, ontology: dict[str, Any]) -> dict[str, Any]:
    out = {
        "nodes": [dict(node) for node in pred_graph.get("nodes", [])],
        "binary_edges": [dict(edge) for edge in pred_graph.get("binary_edges", [])],
        "graph_type": "3d_scene_graph",
        "coordinate_frame": "mujoco_world",
    }
    for node in out["nodes"]:
        idx = ontology["nodes"][node["id"]]["index"]
        node["position_world_xyz"] = [float(x) for x in pred_xyz[idx]]
    return out


def load_model(checkpoint_path: Path, ontology: dict[str, Any], device: str):
    import torch

    from scene_graph_generator.graph_generator.models.depth_augmented import OpenVLAOnlyPooledMLP3DGraphGenerator

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_dim = int(ckpt.get("openvla_dim", 4096))
    model = OpenVLAOnlyPooledMLP3DGraphGenerator(
        input_dim,
        len(ontology["nodes"]),
        len(ontology["predicates"]),
        hidden_dim=1024,
        num_layers=3,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def parse_grid(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def collect_split_arrays(args, split: str, extractor, model, ontology: dict[str, Any]) -> dict[str, Any]:
    import torch
    from PIL import Image

    rows = build_frame_rows(args.data_root, split, frame_stride=1)
    if args.max_frames is not None:
        rows = rows[: args.max_frames]
    if not rows:
        raise RuntimeError(f"No rows for split={split}")

    npz_cache = NpzCache()
    xyz_sidecar_root = args.xyz_sidecar_root or (args.data_root / "inspection" / "graph3d_positions_all")
    xyz_cache = RolloutXyzTargetCache(ontology, data_root=args.data_root, sidecar_root=xyz_sidecar_root)
    node_probs = []
    edge_probs = []
    xyz_pred_all = []
    y_node_np = []
    y_edge_np = []
    y_xyz_np = []
    y_xyz_mask_np = []
    row_order = []
    frame_meta = []
    actions = []

    total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    for batch_idx, start in enumerate(range(0, len(rows), args.batch_size), 1):
        if args.progress_interval and (batch_idx == 1 or batch_idx % args.progress_interval == 0 or batch_idx == total_batches):
            print(f"[{split}] batch {batch_idx}/{total_batches} frames {start}-{min(start + args.batch_size, len(rows))}/{len(rows)}", file=sys.stderr, flush=True)
        batch_rows = rows[start : start + args.batch_size]
        images = []
        instructions = []
        batch_y_node = []
        batch_y_edge = []
        batch_y_xyz = []
        batch_y_xyz_mask = []
        for row in batch_rows:
            arrays = npz_cache.get(str(row["frames_npz"]))
            idx = int(row["frame_index"])
            meta = read_json(Path(row["episode_dir"]) / "metadata.json")
            frames = meta.get("frames") or []
            images.append(Image.fromarray(arrays["rgb"][idx]))
            instructions.append(str(row["instruction"]))
            batch_y_node.append(arrays["node_valid_mask"][idx].astype(np.float32))
            batch_y_edge.append(arrays["oracle_graph_tensor"][idx].astype(np.float32))
            xyz, xyz_mask, _xyz_source = xyz_cache.get(Path(row["episode_dir"]), arrays, idx)
            batch_y_xyz.append(xyz)
            batch_y_xyz_mask.append(xyz_mask)
            frame_meta.append(frames[idx] if idx < len(frames) else {})
            actions.append(arrays["executed_action"][idx].astype(np.float32))
            row_order.append(row)

        features, attn, token_type = extractor.extract_batch(images, instructions, feature_layer=args.feature_layer)
        with torch.inference_mode():
            out = model(features, attn, token_type)
        node_probs.append(sigmoid(out["node_logits"].detach().cpu().float().numpy()))
        edge_probs.append(sigmoid(out["edge_logits"].detach().cpu().float().numpy()))
        xyz_pred_all.append(out["xyz"].detach().cpu().float().numpy())
        y_node_np.extend(batch_y_node)
        y_edge_np.extend(batch_y_edge)
        y_xyz_np.extend(batch_y_xyz)
        y_xyz_mask_np.extend(batch_y_xyz_mask)

    return {
        "split": split,
        "rows": row_order,
        "frame_meta": frame_meta,
        "actions": actions,
        "node_probs": np.concatenate(node_probs),
        "edge_probs": np.concatenate(edge_probs),
        "xyz_pred": np.concatenate(xyz_pred_all),
        "y_node": np.stack(y_node_np).astype(bool),
        "y_edge": np.stack(y_edge_np).astype(bool),
        "xyz_gt": np.stack(y_xyz_np),
        "xyz_mask": np.stack(y_xyz_mask_np),
    }


def select_predicate_thresholds(
    validation_arrays: dict[str, Any],
    ontology: dict[str, Any],
    validity_mask: np.ndarray,
    grid: list[float],
    node_threshold: float,
) -> tuple[dict[str, float], dict[str, Any]]:
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
                summary = score_arrays(
                    validation_arrays,
                    ontology,
                    validity_mask,
                    node_threshold=node_threshold,
                    predicate_thresholds=candidate_thresholds,
                )
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


def add_xyz(summary: dict[str, Any], arrays: dict[str, Any]) -> dict[str, Any]:
    return {**summary, "xyz_metrics": xyz_metrics(arrays["xyz_pred"], arrays["xyz_gt"], arrays["xyz_mask"])}


def summarize_grasp(
    arrays: dict[str, Any],
    ontology: dict[str, Any],
    validity_np: np.ndarray,
    *,
    node_threshold: float,
    predicate_thresholds: dict[str, float],
    output_root: Path,
    save_predictions: bool,
    top_fp: int,
) -> dict[str, Any]:
    grasp_idx = ontology["predicates"]["grasping"]
    examples = []
    schema_errors = 0
    grasp_counts = Counter()
    grasp_fp_by_policy = Counter()
    grasp_fp_by_failure = Counter()
    grasp_fp_by_target = Counter()
    grasp_fp_no_contact = 0
    grasp_fp_no_oracle_grasp = 0
    gripper_action_fp = []
    gripper_action_all = []
    high_conf_fp = []
    split = str(arrays["split"])
    pred_root = output_root / "predictions" / split

    for i, row in enumerate(arrays["rows"]):
        pred_graph = decode_graph(
            np.log(np.clip(arrays["node_probs"][i], 1e-7, 1.0 - 1e-7) / np.clip(1.0 - arrays["node_probs"][i], 1e-7, 1.0)),
            np.log(np.clip(arrays["edge_probs"][i], 1e-7, 1.0 - 1e-7) / np.clip(1.0 - arrays["edge_probs"][i], 1e-7, 1.0)),
            ontology,
            validity_np,
            node_threshold=node_threshold,
            predicate_thresholds=predicate_thresholds,
            include_confidence=True,
        )
        gt_graph = graph_from_targets(arrays["y_node"][i], arrays["y_edge"][i], ontology)
        schema_errors += len(validate_graph(pred_graph))
        examples.append(
            {
                "pred_nodes": graph_node_ids(pred_graph),
                "gt_nodes": graph_node_ids(gt_graph),
                "pred_edges": graph_triplets(pred_graph),
                "gt_edges": graph_triplets(gt_graph),
            }
        )

        pred_grasp = {(e["subject"], e["object"]) for e in pred_graph["binary_edges"] if e["predicate"] == "grasping"}
        gt_grasp = {(e["subject"], e["object"]) for e in gt_graph["binary_edges"] if e["predicate"] == "grasping"}
        fp_set = pred_grasp - gt_grasp
        grasp_counts["tp"] += len(pred_grasp & gt_grasp)
        grasp_counts["fp"] += len(fp_set)
        grasp_counts["fn"] += len(gt_grasp - pred_grasp)
        grasp_counts["frames_with_pred"] += int(bool(pred_grasp))
        grasp_counts["frames_with_gt"] += int(bool(gt_grasp))
        grasp_counts["frames"] += 1

        action = np.asarray(arrays["actions"][i], dtype=np.float32)
        if action.shape[0] >= 7:
            gripper_action_all.append(float(action[6]))
        frame_meta = arrays["frame_meta"][i]
        if fp_set:
            grasp_fp_by_policy[str(row["policy_id"])] += 1
            grasp_fp_by_failure[str(row.get("failure_category", ""))] += 1
            if not bool(frame_meta.get("has_grasping", False)):
                grasp_fp_no_oracle_grasp += 1
            if int(frame_meta.get("contact_count", 0) or 0) == 0:
                grasp_fp_no_contact += 1
            if action.shape[0] >= 7:
                gripper_action_fp.append(float(action[6]))

        for subj, obj in fp_set:
            grasp_fp_by_target[obj] += 1
            s_idx = ontology["nodes"][subj]["index"]
            o_idx = ontology["nodes"][obj]["index"]
            high_conf_fp.append(
                {
                    "prob": float(arrays["edge_probs"][i, s_idx, o_idx, grasp_idx]),
                    "policy_id": row["policy_id"],
                    "failure_category": row.get("failure_category", ""),
                    "episode_id": row["episode_id"],
                    "frame_index": int(row["frame_index"]),
                    "pred": [subj, "grasping", obj],
                    "contact_count": int(frame_meta.get("contact_count", 0) or 0),
                    "oracle_has_grasping": bool(frame_meta.get("has_grasping", False)),
                    "executed_action_gripper_dim": float(action[6]) if action.shape[0] >= 7 else None,
                }
            )
        if save_predictions:
            pred3d = graph_with_xyz(pred_graph, arrays["xyz_pred"][i], ontology)
            pred3d["metadata"] = {
                "split": split,
                "policy_id": row["policy_id"],
                "episode_id": row["episode_id"],
                "frame_index": int(row["frame_index"]),
                "instruction": row["instruction"],
                "gt_grasping": sorted([list(x) for x in gt_grasp]),
                "pred_grasping": sorted([list(x) for x in pred_grasp]),
            }
            write_json(pred_root / str(row["policy_id"]) / str(row["episode_id"]) / f"{int(row['frame_index']):06d}.json", pred3d)

    def stats(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"n": 0, "mean": None, "min": None, "max": None}
        arr = np.asarray(values, dtype=np.float32)
        return {"n": int(arr.size), "mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}

    return {
        "schema_error_count": schema_errors,
        "metrics_from_decoded_graphs": summarize_examples(examples),
        "grasping": {
            **prf(grasp_counts["tp"], grasp_counts["fp"], grasp_counts["fn"]),
            "frames": grasp_counts["frames"],
            "frames_with_pred": grasp_counts["frames_with_pred"],
            "frames_with_gt": grasp_counts["frames_with_gt"],
            "false_positive_frames_or_edges": grasp_counts["fp"],
            "false_positive_no_contact_frames": grasp_fp_no_contact,
            "false_positive_no_oracle_grasp_frames": grasp_fp_no_oracle_grasp,
            "false_positive_by_policy": dict(grasp_fp_by_policy),
            "false_positive_by_failure_category": dict(grasp_fp_by_failure),
            "false_positive_by_target_object": dict(grasp_fp_by_target),
            "executed_action_gripper_dim_all_stats": stats(gripper_action_all),
            "executed_action_gripper_dim_false_positive_stats": stats(gripper_action_fp),
            "high_confidence_false_positives": sorted(high_conf_fp, key=lambda x: x["prob"], reverse=True)[:top_fp],
            "note": "No gripper-closed threshold is assumed; action dim 6 is reported raw to avoid guessing the OpenVLA gripper convention.",
        },
        "predictions": str(pred_root) if save_predictions else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate live Frozen-OpenVLA Graph Generator with grasp diagnostics.")
    parser.add_argument("--data-root", type=Path, default=Path("data/openvla_rollout_graph_v2"))
    parser.add_argument("--ontology", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/ontology/ontology.json"))
    parser.add_argument("--openvla-checkpoint", type=Path, default=Path("checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats"))
    parser.add_argument("--graph-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rollout_depthfree_graph_generator_live_eval"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--feature-layer", type=int, default=-2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--node-threshold", type=float, default=0.5)
    parser.add_argument("--predicate-threshold", type=float, default=0.5)
    parser.add_argument("--grasp-threshold", type=float, default=0.5)
    parser.add_argument("--select-thresholds", action="store_true")
    parser.add_argument(
        "--grid",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--top-fp", type=int, default=50)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--xyz-sidecar-root", type=Path, default=None)
    parser.add_argument("--require-xyz-targets", action="store_true")
    args = parser.parse_args()

    started = time.time()
    args.openvla_checkpoint = resolve_checkpoint(args.openvla_checkpoint)
    ontology = read_json(args.ontology)
    validity_np = relation_validity_mask(ontology)
    grasp_idx = ontology["predicates"]["grasping"]
    idx_to_node = {meta["index"]: node_id for node_id, meta in ontology["nodes"].items()}

    rows = build_frame_rows(args.data_root, args.split, frame_stride=1)
    if args.max_frames is not None:
        rows = rows[: args.max_frames]
    if not rows:
        raise RuntimeError(f"No rows for split={args.split}")

    import torch
    from PIL import Image

    extractor = OpenVLAFeatureExtractor(args.openvla_checkpoint, device=args.device, dtype=args.dtype)
    model, graph_ckpt = load_model(args.graph_checkpoint, ontology, args.device)
    if args.select_thresholds:
        grid = parse_grid(args.grid)
        validation_arrays = collect_split_arrays(args, "validation", extractor, model, ontology)
        eval_arrays = validation_arrays if args.split == "validation" else collect_split_arrays(args, args.split, extractor, model, ontology)
        node_selection = best_binary_threshold(validation_arrays["node_probs"].reshape(-1), validation_arrays["y_node"].reshape(-1), grid)
        node_threshold = float(node_selection["threshold"])
        predicate_thresholds, selected = select_predicate_thresholds(validation_arrays, ontology, validity_np, grid, node_threshold)
        validation_summary = add_xyz(selected["validation"], validation_arrays)
        eval_summary = add_xyz(
            score_arrays(eval_arrays, ontology, validity_np, node_threshold=node_threshold, predicate_thresholds=predicate_thresholds),
            eval_arrays,
        )
        if args.require_xyz_targets and int(eval_summary["xyz_metrics"].get("num_points", 0)) <= 0:
            raise RuntimeError(
                "No valid XYZ targets found for evaluation. "
                f"Checked frames.npz and sidecars under {args.xyz_sidecar_root or (args.data_root / 'inspection' / 'graph3d_positions_all')}."
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
            "selection_split": "validation",
            "eval_split": args.split,
            "num_validation_frames": int(validation_arrays["node_probs"].shape[0]),
            "num_eval_frames": int(eval_arrays["node_probs"].shape[0]),
            "openvla_checkpoint": str(args.openvla_checkpoint),
            "graph_checkpoint": str(args.graph_checkpoint),
            "graph_checkpoint_epoch": graph_ckpt.get("epoch"),
            "xyz_sidecar_root": str(args.xyz_sidecar_root or (args.data_root / "inspection" / "graph3d_positions_all")),
            "openvla_forward_used": True,
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
        write_json(out_path, report)
        lines = [
            "# Rollout Graph Generator Evaluation",
            "",
            f"- selection split: `validation`",
            f"- eval split: `{args.split}`",
            f"- validation frames: `{validation_arrays['node_probs'].shape[0]}`",
            f"- eval frames: `{eval_arrays['node_probs'].shape[0]}`",
            f"- node threshold: `{node_threshold:.2f}`",
            f"- grasp threshold: `{predicate_thresholds.get('grasping', 0.5):.2f}`",
            f"- triplet f1: `{eval_summary['metrics']['triplet']['f1']:.6f}`",
            f"- triplet macro_f1: `{eval_summary['metrics']['triplet']['macro_f1']:.6f}`",
            f"- graph exact match: `{eval_summary['metrics']['graph']['exact_match']:.6f}`",
            f"- grasp precision: `{eval_grasp['grasping']['precision']:.6f}`",
            f"- grasp recall: `{eval_grasp['grasping']['recall']:.6f}`",
            f"- grasp f1: `{eval_grasp['grasping']['f1']:.6f}`",
            f"- grasp fp: `{eval_grasp['grasping']['fp']}`",
            f"- grasp fn: `{eval_grasp['grasping']['fn']}`",
            f"- grasp FP with no contact: `{eval_grasp['grasping']['false_positive_no_contact_frames']}`",
            f"- grasp FP with oracle has_grasping=false: `{eval_grasp['grasping']['false_positive_no_oracle_grasp_frames']}`",
            "",
            "See JSON for predicate thresholds, xyz metrics, high-confidence false positives, and raw gripper action dim statistics.",
        ]
        md_path = args.output_root / "reports" / f"{args.split}_selected_thresholds_grasp_eval.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({"status": "ok", "output": str(out_path), "report": str(md_path)}, sort_keys=True))
        return

    npz_cache = NpzCache()
    xyz_sidecar_root = args.xyz_sidecar_root or (args.data_root / "inspection" / "graph3d_positions_all")
    xyz_cache = RolloutXyzTargetCache(ontology, data_root=args.data_root, sidecar_root=xyz_sidecar_root)

    examples = []
    schema_errors = 0
    xyz_pred_all = []
    xyz_gt_all = []
    xyz_mask_all = []
    grasp_counts = Counter()
    grasp_fp_by_policy = Counter()
    grasp_fp_by_failure = Counter()
    grasp_fp_by_target = Counter()
    grasp_fp_no_contact = 0
    grasp_fp_no_oracle_grasp = 0
    gripper_action_fp = []
    gripper_action_all = []
    high_conf_fp = []

    pred_root = args.output_root / "predictions" / args.split
    total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    for batch_idx, start in enumerate(range(0, len(rows), args.batch_size), 1):
        if args.progress_interval and (batch_idx == 1 or batch_idx % args.progress_interval == 0 or batch_idx == total_batches):
            print(f"[{args.split}] batch {batch_idx}/{total_batches} frames {start}-{min(start + args.batch_size, len(rows))}/{len(rows)}", file=sys.stderr, flush=True)
        batch_rows = rows[start : start + args.batch_size]
        images = []
        instructions = []
        y_node_np = []
        y_edge_np = []
        y_xyz_np = []
        y_xyz_mask_np = []
        frame_meta = []
        actions = []
        for row in batch_rows:
            arrays = npz_cache.get(str(row["frames_npz"]))
            idx = int(row["frame_index"])
            meta = read_json(Path(row["episode_dir"]) / "metadata.json")
            images.append(Image.fromarray(arrays["rgb"][idx]))
            instructions.append(str(row["instruction"]))
            y_node_np.append(arrays["node_valid_mask"][idx].astype(np.float32))
            y_edge_np.append(arrays["oracle_graph_tensor"][idx].astype(np.float32))
            xyz, xyz_mask, _xyz_source = xyz_cache.get(Path(row["episode_dir"]), arrays, idx)
            y_xyz_np.append(xyz)
            y_xyz_mask_np.append(xyz_mask)
            frame_meta.append((meta.get("frames") or [{}])[idx] if idx < len(meta.get("frames") or []) else {})
            actions.append(arrays["executed_action"][idx])

        features, attn, token_type = extractor.extract_batch(images, instructions, feature_layer=args.feature_layer)
        with torch.inference_mode():
            out = model(features, attn, token_type)
        node_logits = out["node_logits"].detach().cpu().float().numpy()
        edge_logits = out["edge_logits"].detach().cpu().float().numpy()
        xyz_pred = out["xyz"].detach().cpu().float().numpy()
        xyz_pred_all.append(xyz_pred)
        xyz_gt_all.append(np.stack(y_xyz_np))
        xyz_mask_all.append(np.stack(y_xyz_mask_np))

        for i, row in enumerate(batch_rows):
            thresholds = {name: args.predicate_threshold for name in ontology["predicates"]}
            thresholds["grasping"] = args.grasp_threshold
            pred_graph = decode_graph(
                node_logits[i],
                edge_logits[i],
                ontology,
                validity_np,
                node_threshold=args.node_threshold,
                predicate_thresholds=thresholds,
                include_confidence=True,
            )
            gt_graph = graph_from_targets(y_node_np[i], y_edge_np[i], ontology)
            schema_errors += len(validate_graph(pred_graph))
            examples.append(
                {
                    "pred_nodes": graph_node_ids(pred_graph),
                    "gt_nodes": graph_node_ids(gt_graph),
                    "pred_edges": graph_triplets(pred_graph),
                    "gt_edges": graph_triplets(gt_graph),
                }
            )

            pred_grasp = {(e["subject"], e["object"]) for e in pred_graph["binary_edges"] if e["predicate"] == "grasping"}
            gt_grasp = {(e["subject"], e["object"]) for e in gt_graph["binary_edges"] if e["predicate"] == "grasping"}
            tp_set = pred_grasp & gt_grasp
            fp_set = pred_grasp - gt_grasp
            fn_set = gt_grasp - pred_grasp
            grasp_counts["tp"] += len(tp_set)
            grasp_counts["fp"] += len(fp_set)
            grasp_counts["fn"] += len(fn_set)
            grasp_counts["frames_with_pred"] += int(bool(pred_grasp))
            grasp_counts["frames_with_gt"] += int(bool(gt_grasp))
            grasp_counts["frames"] += 1
            action = np.asarray(actions[i], dtype=np.float32)
            if action.shape[0] >= 7:
                gripper_action_all.append(float(action[6]))
            if fp_set:
                grasp_fp_by_policy[str(row["policy_id"])] += 1
                grasp_fp_by_failure[str(row.get("failure_category", ""))] += 1
                if not bool(frame_meta[i].get("has_grasping", False)):
                    grasp_fp_no_oracle_grasp += 1
                if int(frame_meta[i].get("contact_count", 0) or 0) == 0:
                    grasp_fp_no_contact += 1
                if action.shape[0] >= 7:
                    gripper_action_fp.append(float(action[6]))
            edge_prob = sigmoid(edge_logits[i])
            for subj, obj in fp_set:
                grasp_fp_by_target[obj] += 1
                s_idx = ontology["nodes"][subj]["index"]
                o_idx = ontology["nodes"][obj]["index"]
                high_conf_fp.append(
                    {
                        "prob": float(edge_prob[s_idx, o_idx, grasp_idx]),
                        "policy_id": row["policy_id"],
                        "failure_category": row.get("failure_category", ""),
                        "episode_id": row["episode_id"],
                        "frame_index": int(row["frame_index"]),
                        "pred": [subj, "grasping", obj],
                        "contact_count": int(frame_meta[i].get("contact_count", 0) or 0),
                        "oracle_has_grasping": bool(frame_meta[i].get("has_grasping", False)),
                        "executed_action_gripper_dim": float(action[6]) if action.shape[0] >= 7 else None,
                    }
                )
            if args.save_predictions:
                pred3d = graph_with_xyz(pred_graph, xyz_pred[i], ontology)
                pred3d["metadata"] = {
                    "split": args.split,
                    "policy_id": row["policy_id"],
                    "episode_id": row["episode_id"],
                    "frame_index": int(row["frame_index"]),
                    "instruction": row["instruction"],
                    "gt_grasping": sorted([list(x) for x in gt_grasp]),
                    "pred_grasping": sorted([list(x) for x in pred_grasp]),
                }
                write_json(pred_root / str(row["policy_id"]) / str(row["episode_id"]) / f"{int(row['frame_index']):06d}.json", pred3d)

    metrics = summarize_examples(examples)
    grasp_prf = prf(grasp_counts["tp"], grasp_counts["fp"], grasp_counts["fn"])
    high_conf_fp = sorted(high_conf_fp, key=lambda x: x["prob"], reverse=True)[: args.top_fp]

    def stats(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"n": 0, "mean": None, "min": None, "max": None}
        arr = np.asarray(values, dtype=np.float32)
        return {"n": int(arr.size), "mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}

    report = {
        "status": "ok",
        "split": args.split,
        "num_frames": len(rows),
        "openvla_checkpoint": str(args.openvla_checkpoint),
        "graph_checkpoint": str(args.graph_checkpoint),
        "graph_checkpoint_epoch": graph_ckpt.get("epoch"),
        "xyz_sidecar_root": str(xyz_sidecar_root),
        "thresholds": {
            "node": args.node_threshold,
            "predicate_default": args.predicate_threshold,
            "grasping": args.grasp_threshold,
        },
        "schema_error_count": schema_errors,
        "metrics": metrics,
        "xyz_metrics": xyz_metrics(np.concatenate(xyz_pred_all), np.concatenate(xyz_gt_all), np.concatenate(xyz_mask_all)),
        "grasping": {
            **grasp_prf,
            "frames": grasp_counts["frames"],
            "frames_with_pred": grasp_counts["frames_with_pred"],
            "frames_with_gt": grasp_counts["frames_with_gt"],
            "false_positive_frames_or_edges": grasp_counts["fp"],
            "false_positive_no_contact_frames": grasp_fp_no_contact,
            "false_positive_no_oracle_grasp_frames": grasp_fp_no_oracle_grasp,
            "false_positive_by_policy": dict(grasp_fp_by_policy),
            "false_positive_by_failure_category": dict(grasp_fp_by_failure),
            "false_positive_by_target_object": dict(grasp_fp_by_target),
            "executed_action_gripper_dim_all_stats": stats(gripper_action_all),
            "executed_action_gripper_dim_false_positive_stats": stats(gripper_action_fp),
            "high_confidence_false_positives": high_conf_fp,
            "note": "No gripper-closed threshold is assumed; action dim 6 is reported raw to avoid guessing the OpenVLA gripper convention.",
        },
        "predictions": str(pred_root) if args.save_predictions else None,
        "elapsed_sec": round(time.time() - started, 3),
    }
    if args.require_xyz_targets and int(report["xyz_metrics"].get("num_points", 0)) <= 0:
        raise RuntimeError(
            "No valid XYZ targets found for evaluation. "
            f"Checked frames.npz and sidecars under {xyz_sidecar_root}."
        )
    out_path = args.output_root / "metrics" / f"{args.split}_grasp_eval.json"
    write_json(out_path, report)
    lines = [
        "# Rollout Graph Generator Evaluation",
        "",
        f"- split: `{args.split}`",
        f"- frames: `{len(rows)}`",
        f"- graph checkpoint: `{args.graph_checkpoint}`",
        f"- triplet f1: `{metrics['triplet']['f1']:.6f}`",
        f"- triplet macro_f1: `{metrics['triplet']['macro_f1']:.6f}`",
        f"- grasp precision: `{grasp_prf['precision']:.6f}`",
        f"- grasp recall: `{grasp_prf['recall']:.6f}`",
        f"- grasp f1: `{grasp_prf['f1']:.6f}`",
        f"- grasp fp: `{grasp_counts['fp']}`",
        f"- grasp fn: `{grasp_counts['fn']}`",
        f"- grasp FP with no contact: `{grasp_fp_no_contact}`",
        f"- grasp FP with oracle has_grasping=false: `{grasp_fp_no_oracle_grasp}`",
        "",
        "See JSON for high-confidence false positives and gripper action dim statistics.",
    ]
    md_path = args.output_root / "reports" / f"{args.split}_grasp_eval.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(out_path), "report": str(md_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
