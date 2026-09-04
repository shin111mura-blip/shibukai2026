#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()


def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def prf(pred: set, gold: set) -> dict:
    tp = len(pred & gold)
    precision = 0.0 if not pred else tp / len(pred)
    recall = 0.0 if not gold else tp / len(gold)
    return {"precision": precision, "recall": recall, "f1": f1(precision, recall), "tp": tp, "pred": len(pred), "gold": len(gold)}


def graph_sets(graph: dict) -> tuple[set, set]:
    nodes = {node["id"] for node in graph.get("nodes", [])}
    edges = {(edge["subject"], edge["predicate"], edge["object"]) for edge in graph.get("binary_edges", [])}
    return nodes, edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-id", default="demo_0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_gt_v0"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_path = args.output_dir / "reports" / "selected_frames.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected_frames"] if selected_path.exists() else []
    frames = [item["frame_id"] for item in selected]
    invalid = 0
    node_scores = []
    edge_scores = []
    relation_gold = defaultdict(set)
    relation_pred = defaultdict(set)
    exact = 0
    for frame_id in frames:
        gold_path = args.output_dir / "rule_based" / "observable_graph" / args.demo_id / f"{frame_id:06d}.json"
        pred_path = args.output_dir / "qwen_zero_shot" / "parsed" / args.demo_id / f"{frame_id:06d}.json"
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        if pred_path.exists():
            pred = json.loads(pred_path.read_text(encoding="utf-8"))
        else:
            invalid += 1
            pred = {"nodes": [], "binary_edges": []}
        gold_nodes, gold_edges = graph_sets(gold)
        pred_nodes, pred_edges = graph_sets(pred)
        node_scores.append(prf(pred_nodes, gold_nodes))
        edge_scores.append(prf(pred_edges, gold_edges))
        exact += int(gold_nodes == pred_nodes and gold_edges == pred_edges)
        for edge in gold_edges:
            relation_gold[edge[1]].add((frame_id,) + edge)
        for edge in pred_edges:
            relation_pred[edge[1]].add((frame_id,) + edge)
    def avg(items: list[dict], key: str) -> float:
        return 0.0 if not items else sum(item[key] for item in items) / len(items)
    relation_scores = {rel: prf(relation_pred[rel], relation_gold[rel]) for rel in sorted(set(relation_gold) | set(relation_pred))}
    payload = {
        "frames": frames,
        "invalid_json_rate": 0.0 if not frames else invalid / len(frames),
        "node": {"precision": avg(node_scores, "precision"), "recall": avg(node_scores, "recall"), "f1": avg(node_scores, "f1"), "exact_match": avg([{"exact": 1.0 if s["pred"] == s["gold"] == s["tp"] else 0.0} for s in node_scores], "exact")},
        "edge": {"precision": avg(edge_scores, "precision"), "recall": avg(edge_scores, "recall"), "f1": avg(edge_scores, "f1")},
        "relation_wise": relation_scores,
        "grasping": relation_scores.get("grasping", {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "pred": 0, "gold": 0}),
        "exact_graph_match": 0.0 if not frames else exact / len(frames),
    }
    out = args.output_dir / "comparison" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
