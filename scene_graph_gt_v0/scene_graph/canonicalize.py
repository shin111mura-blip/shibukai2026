from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List

from .schema import Edge, Node, RELATION_RULE_VERSION, SCHEMA_VERSION, validate_edges, validate_graph, validate_nodes


def canonical_nodes(nodes: Iterable[Node]) -> List[Node]:
    dedup = {node.id: node for node in nodes}
    ordered = [dedup[key] for key in sorted(dedup)]
    validate_nodes(ordered)
    return ordered


def canonical_edges(edges: Iterable[Edge], node_ids: Iterable[str]) -> List[Edge]:
    dedup = {(edge.subject, edge.predicate, edge.object): edge for edge in edges}
    ordered = [dedup[key] for key in sorted(dedup)]
    validate_edges(ordered, node_ids)
    return ordered


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_payload(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def make_config_hash(config: Dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(config).encode("utf-8")).hexdigest()


def make_graph(
    *,
    source: str,
    mode: str,
    task_id: str,
    demo_id: str,
    frame_id: int,
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    config_hash: str,
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    ordered_nodes = canonical_nodes(nodes)
    ordered_edges = canonical_edges(edges, [node.id for node in ordered_nodes])
    metadata = {"relation_rule_version": RELATION_RULE_VERSION, "config_hash": config_hash}
    if extra_metadata:
        metadata.update(extra_metadata)
    graph = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "mode": mode,
        "task_id": task_id,
        "demo_id": demo_id,
        "frame_id": int(frame_id),
        "nodes": [node.to_json(include_geometry=False) for node in ordered_nodes],
        "binary_edges": [edge.to_json() for edge in ordered_edges],
        "metadata": metadata,
    }
    validate_graph(graph)
    return graph
