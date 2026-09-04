from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_graph(
    node_logits,
    edge_logits,
    ontology: Dict[str, Any],
    validity_mask,
    *,
    node_threshold: float = 0.5,
    predicate_thresholds: Mapping[str, float] | None = None,
    include_confidence: bool = False,
) -> Dict[str, Any]:
    node_prob = sigmoid(np.asarray(node_logits))
    edge_prob = sigmoid(np.asarray(edge_logits))
    idx_to_node = {meta["index"]: (node_id, meta) for node_id, meta in ontology["nodes"].items()}
    idx_to_pred = {idx: pred for pred, idx in ontology["predicates"].items()}
    pred_nodes_bool = node_prob >= node_threshold
    nodes = []
    for idx, present in enumerate(pred_nodes_bool):
        if present:
            node_id, meta = idx_to_node[idx]
            node = {"id": node_id, "category": meta["category"], "entity_type": meta["entity_type"], "present": True}
            if include_confidence:
                node["confidence"] = float(node_prob[idx])
            nodes.append(node)
    edges = []
    thresholds = predicate_thresholds or {}
    for i in range(edge_prob.shape[0]):
        if not pred_nodes_bool[i]:
            continue
        for j in range(edge_prob.shape[1]):
            if not pred_nodes_bool[j]:
                continue
            for r in range(edge_prob.shape[2]):
                pred = idx_to_pred[r]
                threshold = thresholds.get(pred, 0.5)
                if validity_mask[i, j, r] and edge_prob[i, j, r] >= threshold:
                    edge = {"subject": idx_to_node[i][0], "predicate": pred, "object": idx_to_node[j][0]}
                    if include_confidence:
                        edge["confidence"] = float(edge_prob[i, j, r])
                        edge["threshold"] = float(threshold)
                    edges.append(edge)
    return {"nodes": sorted(nodes, key=lambda x: x["id"]), "binary_edges": sorted(edges, key=lambda x: (x["subject"], x["predicate"], x["object"]))}
