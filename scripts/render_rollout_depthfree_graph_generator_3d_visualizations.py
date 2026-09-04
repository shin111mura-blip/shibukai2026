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

from scene_graph_generator.graph_generator.decoding import decode_graph
from scene_graph_generator.graph_generator.feature_extractor import OpenVLAFeatureExtractor
from scene_graph_generator.graph_generator.masks import relation_validity_mask
from scene_graph_generator.graph_generator.schema import graph_triplets, read_json, validate_graph, write_json
from scene_graph_generator.scripts.render_depth_3d_visualizations import common_bounds, coordinate_errors, draw_graph_3d
from scripts.evaluate_rollout_depthfree_graph_generator_live import graph_with_xyz, load_model
from scripts.rollout_xyz_targets import RolloutXyzTargetCache
from scripts.train_rollout_depthfree_graph_generator_live import NpzCache, build_frame_rows, graph_from_targets, resolve_checkpoint


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def add_xyz_to_gt(gt_graph: dict[str, Any], xyz: np.ndarray, xyz_mask: np.ndarray, ontology: dict[str, Any]) -> dict[str, Any]:
    out = {
        "nodes": [dict(node) for node in gt_graph.get("nodes", [])],
        "binary_edges": [dict(edge) for edge in gt_graph.get("binary_edges", [])],
        "graph_type": "3d_scene_graph",
        "coordinate_frame": "mujoco_world",
    }
    for node in out["nodes"]:
        idx = ontology["nodes"][node["id"]]["index"]
        if bool(xyz_mask[idx]):
            node["position_world_xyz"] = [float(v) for v in xyz[idx]]
    return out


def has_grasping(graph: dict[str, Any]) -> bool:
    return any(edge.get("predicate") == "grasping" for edge in graph.get("binary_edges", []))


def render_pair(gt: dict[str, Any], pred: dict[str, Any], title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 6), dpi=160)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    bounds = common_bounds([gt, pred])
    draw_graph_3d(ax1, gt, "GT 3D Scene Graph", bounds=bounds)
    draw_graph_3d(ax2, pred, "Predicted 3D Scene Graph", bounds=bounds)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def select_diverse_rows(rows: list[dict[str, Any]], max_scan: int) -> list[dict[str, Any]]:
    by_task: dict[int, list[dict[str, Any]]] = {}
    for row in rows[:max_scan]:
        by_task.setdefault(int(row["task_id"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        task_rows = by_task[task_id]
        if task_rows:
            selected.append(task_rows[min(len(task_rows) // 2, len(task_rows) - 1)])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Render GT vs predicted 3D graph pairs for the rollout-retrained GraphGenerator.")
    parser.add_argument("--data-root", type=Path, default=Path("data/openvla_rollout_graph_v2"))
    parser.add_argument("--ontology", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/ontology/ontology.json"))
    parser.add_argument("--openvla-checkpoint", type=Path, default=Path("checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats"))
    parser.add_argument(
        "--graph-checkpoint",
        type=Path,
        default=Path("outputs/rollout_depthfree_graph_generator_live_v1/checkpoints/rollout_live_depthfree_pooled_mlp_3d/best.pt"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rollout_depthfree_graph_generator_live_eval_corrected_visualizations"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--feature-layer", type=int, default=-2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--node-threshold", type=float, default=0.5)
    parser.add_argument("--predicate-threshold", type=float, default=0.5)
    parser.add_argument("--grasp-threshold", type=float, default=0.5)
    parser.add_argument("--thresholds-json", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--max-scan-frames", type=int, default=800)
    parser.add_argument("--xyz-sidecar-root", type=Path, default=None)
    parser.add_argument("--require-xyz-targets", action="store_true")
    args = parser.parse_args()

    import torch
    from PIL import Image

    ontology = read_json(args.ontology)
    validity_np = relation_validity_mask(ontology)
    rows = build_frame_rows(args.data_root, args.split, frame_stride=1)
    rows = select_diverse_rows(rows, args.max_scan_frames)
    if not rows:
        raise RuntimeError(f"No rows selected for split={args.split}")

    args.openvla_checkpoint = resolve_checkpoint(args.openvla_checkpoint)
    extractor = OpenVLAFeatureExtractor(args.openvla_checkpoint, device=args.device, dtype=args.dtype)
    model, ckpt = load_model(args.graph_checkpoint, ontology, args.device)

    npz_cache = NpzCache()
    xyz_sidecar_root = args.xyz_sidecar_root or (args.data_root / "inspection" / "graph3d_positions_all")
    xyz_cache = RolloutXyzTargetCache(ontology, data_root=args.data_root, sidecar_root=xyz_sidecar_root)
    thresholds = {name: args.predicate_threshold for name in ontology["predicates"]}
    thresholds["grasping"] = args.grasp_threshold
    node_threshold = args.node_threshold
    if args.thresholds_json is not None:
        threshold_report = read_json(args.thresholds_json)
        selection = threshold_report.get("threshold_selection") or threshold_report
        node_threshold = float(selection.get("node_threshold", node_threshold))
        thresholds.update({name: float(value) for name, value in (selection.get("predicate_thresholds") or {}).items()})
    out_dir = args.output_root / "visualizations" / args.split / "graph_3d"
    pred_dir = args.output_root / "predictions" / args.split
    gt_dir = args.output_root / "gt" / args.split

    examples = []
    used_kinds: set[str] = set()
    rendered = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        images = []
        instructions = []
        targets = []
        for row in batch:
            arrays = npz_cache.get(str(row["frames_npz"]))
            idx = int(row["frame_index"])
            xyz, xyz_mask, xyz_source = xyz_cache.get(Path(row["episode_dir"]), arrays, idx)
            images.append(Image.fromarray(arrays["rgb"][idx]))
            instructions.append(str(row["instruction"]))
            targets.append((arrays["node_valid_mask"][idx].astype(np.float32), arrays["oracle_graph_tensor"][idx].astype(np.float32), xyz, xyz_mask, xyz_source))
        features, attn, token_type = extractor.extract_batch(images, instructions, feature_layer=args.feature_layer)
        with torch.inference_mode():
            out = model(features, attn, token_type)
        node_logits = out["node_logits"].detach().cpu().float().numpy()
        edge_logits = out["edge_logits"].detach().cpu().float().numpy()
        xyz_pred = out["xyz"].detach().cpu().float().numpy()

        for i, row in enumerate(batch):
            y_node, y_edge, xyz_gt, xyz_mask, xyz_source = targets[i]
            pred_graph = decode_graph(
                node_logits[i],
                edge_logits[i],
                ontology,
                validity_np,
                node_threshold=node_threshold,
                predicate_thresholds=thresholds,
                include_confidence=True,
            )
            gt_graph = graph_from_targets(y_node, y_edge, ontology)
            gt3d = add_xyz_to_gt(gt_graph, xyz_gt, xyz_mask, ontology)
            pred3d = graph_with_xyz(pred_graph, xyz_pred[i], ontology)
            gt_edges = set(graph_triplets(gt3d))
            pred_edges = set(graph_triplets(pred3d))
            gt_grasp = has_grasping(gt3d)
            pred_grasp = has_grasping(pred3d)
            if gt_grasp and pred_grasp:
                kind = "grasp_tp"
            elif pred_grasp and not gt_grasp:
                kind = "grasp_fp"
            elif gt_grasp and not pred_grasp:
                kind = "grasp_fn"
            else:
                kind = "regular"
            idx = rendered
            stem = f"{idx:02d}_{kind}_task_{int(row['task_id']):02d}_{row['policy_id']}_{row['episode_id']}_frame_{int(row['frame_index']):06d}"
            title = (
                f"split={args.split} task={int(row['task_id']):02d} policy={row['policy_id']} "
                f"episode={row['episode_id']} frame={int(row['frame_index']):06d}"
            )
            image_path = out_dir / f"{stem}.png"
            render_pair(gt3d, pred3d, title, image_path)
            gt_path = gt_dir / f"{stem}.json"
            pred_path = pred_dir / f"{stem}.json"
            write_json(gt_path, gt3d)
            write_json(pred_path, pred3d)
            examples.append(
                {
                    "kind": kind,
                    "image": str(image_path),
                    "gt_graph": str(gt_path),
                    "pred_graph": str(pred_path),
                    "split": args.split,
                    "task_id": int(row["task_id"]),
                    "policy_id": row["policy_id"],
                    "episode_id": row["episode_id"],
                    "frame_index": int(row["frame_index"]),
                    "gt_edges": len(gt_edges),
                    "pred_edges": len(pred_edges),
                    "tp_edges": len(gt_edges & pred_edges),
                    "fp_edges": len(pred_edges - gt_edges),
                    "fn_edges": len(gt_edges - pred_edges),
                    "gt_grasping": sorted([list(edge) for edge in gt_edges if edge[1] == "grasping"]),
                    "pred_grasping": sorted([list(edge) for edge in pred_edges if edge[1] == "grasping"]),
                    "coordinate_errors": coordinate_errors(gt3d, pred3d),
                    "schema_errors": validate_graph(pred3d),
                    "xyz_source": xyz_source,
                    "max_edge_probability": float(sigmoid(edge_logits[i]).max()),
                }
            )
            used_kinds.add(kind)
            rendered += 1
            if rendered >= args.max_examples:
                break
        if rendered >= args.max_examples:
            break

    manifest = {
        "status": "ok",
        "split": args.split,
        "graph_checkpoint": str(args.graph_checkpoint),
        "graph_checkpoint_epoch": ckpt.get("epoch"),
        "openvla_checkpoint": str(args.openvla_checkpoint),
        "xyz_sidecar_root": str(xyz_sidecar_root),
        "thresholds": {
            "node": node_threshold,
            "predicate_default": args.predicate_threshold,
            "grasping": args.grasp_threshold,
            "predicate_thresholds": thresholds,
            "source": str(args.thresholds_json) if args.thresholds_json is not None else None,
        },
        "format": "Two-panel 3D comparison: left=GT 3D Scene Graph, right=Predicted 3D Scene Graph.",
        "dir": str(out_dir),
        "examples": examples,
    }
    if args.require_xyz_targets and not any((ex["coordinate_errors"].get("per_node") or {}) for ex in examples):
        raise RuntimeError(
            "No valid XYZ targets found in rendered examples. "
            f"Checked frames.npz and sidecars under {xyz_sidecar_root}."
        )
    write_json(out_dir / "graph_3d_manifest.json", manifest)
    print(json.dumps({"status": "ok", "dir": str(out_dir), "count": len(examples)}, sort_keys=True))


if __name__ == "__main__":
    main()
