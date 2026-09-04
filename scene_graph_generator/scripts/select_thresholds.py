#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .cached_eval_common import collect_probability_records, write_markdown_summary
    from .common import DEFAULT_OUTPUT_ROOT
except ImportError:  # pragma: no cover - CLI path execution
    from cached_eval_common import collect_probability_records, write_markdown_summary
    from common import DEFAULT_OUTPUT_ROOT
from scene_graph_generator.graph_generator.metrics import prf
from scene_graph_generator.graph_generator.schema import write_json


def best_binary_threshold(scores: np.ndarray, targets: np.ndarray, grid: list[float]) -> dict:
    best = None
    target_bool = targets.astype(bool)
    for threshold in grid:
        pred_bool = scores >= threshold
        tp = int(np.logical_and(pred_bool, target_bool).sum())
        fp = int(np.logical_and(pred_bool, ~target_bool).sum())
        fn = int(np.logical_and(~pred_bool, target_bool).sum())
        row = {**prf(tp, fp, fn), "threshold": threshold}
        key = (row["f1"], row["precision"], row["recall"], -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, row)
    return best[1]


def metric_key(summary: dict) -> tuple:
    metrics = summary["metrics"]
    return (
        metrics["triplet"].get("macro_f1", 0.0),
        metrics["triplet"].get("f1", 0.0),
        metrics["graph"].get("exact_match", 0.0),
        metrics["node"].get("f1", 0.0),
        -metrics["graph"].get("normalized_hamming_distance", 1.0),
    )


def make_arrays(records: list[dict]) -> dict:
    return {
        "node_probs": np.stack([r["node_probs"] for r in records]),
        "edge_probs": np.stack([r["edge_probs"] for r in records]),
        "y_node": np.stack([r["row"]["y_node_np"] for r in records]).astype(bool),
        "y_edge": np.stack([r["row"]["y_edge_np"] for r in records]).astype(bool),
    }


def score_arrays(arrays: dict, ontology: dict, validity_mask: np.ndarray, *, node_threshold: float, predicate_thresholds: dict[str, float]) -> dict:
    node_pred = arrays["node_probs"] >= node_threshold
    node_gt = arrays["y_node"]
    thresholds = np.zeros((len(ontology["predicates"]),), dtype=np.float32)
    for name, idx in ontology["predicates"].items():
        thresholds[idx] = predicate_thresholds.get(name, 0.5)
    edge_pred = arrays["edge_probs"] >= thresholds.reshape(1, 1, 1, -1)
    edge_pred &= validity_mask.reshape(1, *validity_mask.shape)
    edge_pred &= node_pred[:, :, None, None]
    edge_pred &= node_pred[:, None, :, None]
    edge_gt = arrays["y_edge"] & validity_mask.reshape(1, *validity_mask.shape)

    def counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int]:
        return (
            int(np.logical_and(pred, gt).sum()),
            int(np.logical_and(pred, ~gt).sum()),
            int(np.logical_and(~pred, gt).sum()),
        )

    node_tp, node_fp, node_fn = counts(node_pred, node_gt)
    edge_tp, edge_fp, edge_fn = counts(edge_pred, edge_gt)
    node_exact = np.equal(node_pred, node_gt).all(axis=1)
    edge_exact = np.equal(edge_pred, edge_gt).reshape(edge_pred.shape[0], -1).all(axis=1)
    graph_exact = node_exact & edge_exact
    node_union = np.logical_or(node_pred, node_gt).sum(axis=1)
    edge_union = np.logical_or(edge_pred, edge_gt).reshape(edge_pred.shape[0], -1).sum(axis=1)
    node_diff = np.logical_xor(node_pred, node_gt).sum(axis=1)
    edge_diff = np.logical_xor(edge_pred, edge_gt).reshape(edge_pred.shape[0], -1).sum(axis=1)
    union = node_union + edge_union
    diff = node_diff + edge_diff
    jaccard = np.divide(union - diff, union, out=np.ones_like(union, dtype=np.float64), where=union != 0)
    hamming = np.divide(diff, union, out=np.zeros_like(union, dtype=np.float64), where=union != 0)

    pred_any = edge_pred.any(axis=0)
    gt_any = edge_gt.any(axis=0)
    idx_to_pred = {idx: pred for pred, idx in ontology["predicates"].items()}
    predicate_metrics = {}
    for idx, name in sorted(idx_to_pred.items()):
        pred = pred_any[:, :, idx]
        gt = gt_any[:, :, idx]
        if not pred.any() and not gt.any():
            continue
        predicate_metrics[name] = prf(*counts(pred, gt))
    macro = sum(row["f1"] for row in predicate_metrics.values()) / len(predicate_metrics) if predicate_metrics else 0.0
    n = int(node_pred.shape[0])
    return {
        "schema_error_count": 0,
        "metrics": {
            "num_examples": n,
            "node": {**prf(node_tp, node_fp, node_fn), "exact_match": float(node_exact.mean()) if n else 0.0},
            "triplet": {**prf(edge_tp, edge_fp, edge_fn), "macro_f1": macro, "exact_match": float(edge_exact.mean()) if n else 0.0},
            "graph": {
                "exact_match": float(graph_exact.mean()) if n else 0.0,
                "jaccard_similarity": float(jaccard.mean()) if n else 0.0,
                "normalized_hamming_distance": float(hamming.mean()) if n else 0.0,
            },
            "predicate": predicate_metrics,
            "macro_zero_positive_policy": "Predicates with zero GT and zero predictions are omitted from macro-F1.",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "feature_cache" / "all_frames")
    ap.add_argument("--architecture", choices=["pooled_mlp", "node_query_decoder"], default="pooled_mlp")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args()

    started = time.time()
    grid = [float(x) for x in args.grid.split(",") if x.strip()]
    bundle = collect_probability_records(
        architecture=args.architecture,
        cache_dir=args.cache_dir,
        output_root=args.output_root,
        split="validation",
        batch_size=args.batch_size,
        device=args.device,
        max_examples=args.max_examples,
    )
    ontology = bundle["ontology"]
    records = bundle["records"]
    validity_mask = bundle["validity_mask"]
    arrays = make_arrays(records)
    node_scores = np.concatenate([r["node_probs"].reshape(-1) for r in records])
    node_targets = np.concatenate([r["row"]["y_node_np"].reshape(-1) for r in records])
    node_result = best_binary_threshold(node_scores, node_targets, grid)
    node_threshold = float(node_result["threshold"])

    predicate_names = sorted(ontology["predicates"])
    global_candidates = []
    for threshold in grid:
        thresholds = {pred: threshold for pred in predicate_names}
        summary = score_arrays(arrays, ontology, validity_mask, node_threshold=node_threshold, predicate_thresholds=thresholds)
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
                summary = score_arrays(arrays, ontology, validity_mask, node_threshold=node_threshold, predicate_thresholds=candidate_thresholds)
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
    thresholds = {
        "status": "ok",
        "architecture": args.architecture,
        "selection_split": "validation",
        "checkpoint": bundle["checkpoint_path"],
        "checkpoint_epoch": bundle["checkpoint_epoch"],
        "node_threshold": node_threshold,
        "node_selection": node_result,
        "predicate_thresholds": pred_thresholds,
        "predicate_selection": predicate_selection,
        "candidate_grid": grid,
        "validation": validation,
        "elapsed_sec": round(time.time() - started, 3),
    }
    out_path = args.output_root / "metrics" / args.architecture / "selected_thresholds.json"
    write_json(out_path, thresholds)
    write_markdown_summary(args.output_root / "reports" / f"{args.architecture}_threshold_selection.md", f"{args.architecture} Threshold Selection", validation)
    print(json.dumps({"status": "ok", "architecture": args.architecture, "output": str(out_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
