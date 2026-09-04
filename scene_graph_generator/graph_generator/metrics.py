from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Set, Tuple


Triplet = Tuple[str, str, str]


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def set_counts(pred: Set, gt: Set) -> Tuple[int, int, int]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    return tp, fp, fn


def jaccard(pred: Set, gt: Set) -> float:
    denom = len(pred | gt)
    return len(pred & gt) / denom if denom else 1.0


def summarize_examples(rows: Iterable[Mapping]) -> Dict:
    totals = defaultdict(int)
    pred_by_predicate: MutableMapping[str, Set[Triplet]] = defaultdict(set)
    gt_by_predicate: MutableMapping[str, Set[Triplet]] = defaultdict(set)
    node_exact = edge_exact = graph_exact = 0
    jac_sum = hamming_sum = 0.0
    n = 0
    for row in rows:
        pred_nodes = set(row["pred_nodes"])
        gt_nodes = set(row["gt_nodes"])
        pred_edges = set(tuple(x) for x in row["pred_edges"])
        gt_edges = set(tuple(x) for x in row["gt_edges"])
        tp, fp, fn = set_counts(pred_nodes, gt_nodes)
        totals["node_tp"] += tp
        totals["node_fp"] += fp
        totals["node_fn"] += fn
        tp, fp, fn = set_counts(pred_edges, gt_edges)
        totals["edge_tp"] += tp
        totals["edge_fp"] += fp
        totals["edge_fn"] += fn
        node_exact += int(pred_nodes == gt_nodes)
        edge_exact += int(pred_edges == gt_edges)
        graph_exact += int(pred_nodes == gt_nodes and pred_edges == gt_edges)
        jac_sum += jaccard(pred_nodes | pred_edges, gt_nodes | gt_edges)
        union_size = len(pred_nodes | gt_nodes) + len(pred_edges | gt_edges)
        diff_size = len(pred_nodes ^ gt_nodes) + len(pred_edges ^ gt_edges)
        hamming_sum += diff_size / union_size if union_size else 0.0
        for edge in pred_edges:
            pred_by_predicate[edge[1]].add(edge)
        for edge in gt_edges:
            gt_by_predicate[edge[1]].add(edge)
        n += 1
    predicates = sorted(set(pred_by_predicate) | set(gt_by_predicate))
    predicate_metrics = {}
    for pred in predicates:
        predicate_metrics[pred] = prf(*set_counts(pred_by_predicate[pred], gt_by_predicate[pred]))
    macro = sum(m["f1"] for m in predicate_metrics.values()) / len(predicate_metrics) if predicate_metrics else 0.0
    return {
        "num_examples": n,
        "node": {**prf(totals["node_tp"], totals["node_fp"], totals["node_fn"]), "exact_match": node_exact / n if n else 0.0},
        "triplet": {
            **prf(totals["edge_tp"], totals["edge_fp"], totals["edge_fn"]),
            "macro_f1": macro,
            "exact_match": edge_exact / n if n else 0.0,
        },
        "graph": {
            "exact_match": graph_exact / n if n else 0.0,
            "jaccard_similarity": jac_sum / n if n else 0.0,
            "normalized_hamming_distance": hamming_sum / n if n else 0.0,
        },
        "predicate": predicate_metrics,
        "macro_zero_positive_policy": "Predicates with zero GT and zero predictions are omitted from macro-F1.",
    }

