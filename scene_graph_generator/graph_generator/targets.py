from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def encode_targets(graph: Dict[str, Any], ontology: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    nodes = ontology["nodes"]
    preds = ontology["predicates"]
    k = len(nodes)
    r = len(preds)
    y_node = np.zeros((k,), dtype=np.float32)
    y_edge = np.zeros((k, k, r), dtype=np.float32)
    for node in graph.get("nodes", []):
        if node["id"] not in nodes:
            raise KeyError(f"Node {node['id']} not in ontology")
        y_node[nodes[node["id"]]["index"]] = 1.0
    for edge in graph.get("binary_edges", []):
        if edge["subject"] not in nodes:
            raise KeyError(f"Subject {edge['subject']} not in ontology")
        if edge["object"] not in nodes:
            raise KeyError(f"Object {edge['object']} not in ontology")
        if edge["predicate"] not in preds:
            raise KeyError(f"Predicate {edge['predicate']} not in ontology")
        i = nodes[edge["subject"]]["index"]
        j = nodes[edge["object"]]["index"]
        p = preds[edge["predicate"]]
        if i != j:
            y_edge[i, j, p] = 1.0
    return y_node, y_edge


def train_positive_counts(graphs, ontology: Dict[str, Any]) -> Dict[str, Any]:
    node_counts = np.zeros((len(ontology["nodes"]),), dtype=np.int64)
    edge_counts = np.zeros((len(ontology["predicates"]),), dtype=np.int64)
    n = 0
    for graph in graphs:
        y_node, y_edge = encode_targets(graph, ontology)
        node_counts += y_node.astype(np.int64)
        edge_counts += y_edge.sum(axis=(0, 1)).astype(np.int64)
        n += 1
    idx_to_node = {meta["index"]: node_id for node_id, meta in ontology["nodes"].items()}
    idx_to_pred = {idx: pred for pred, idx in ontology["predicates"].items()}
    return {
        "num_frames": n,
        "node_positive_counts": {idx_to_node[i]: int(v) for i, v in enumerate(node_counts)},
        "predicate_positive_counts": {idx_to_pred[i]: int(v) for i, v in enumerate(edge_counts)},
    }

