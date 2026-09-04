from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Tuple

from .canonicalize import make_graph
from .schema import ALLOWED_PREDICATES, Edge, Node, validate_graph


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        return json.loads(stripped[start : end + 1])


def parse_qwen_graph(
    *,
    raw_text: str,
    task_id: str,
    demo_id: str,
    frame_id: int,
    allowed_node_ids: Iterable[str],
    config_hash: str,
) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
    parse_info: Dict[str, Any] = {"invalid": False, "errors": [], "rejected_nodes": [], "rejected_edges": []}
    allowed = set(allowed_node_ids)
    try:
        parsed = extract_json_object(raw_text)
        if parsed.get("invalid") is True:
            raise ValueError(f"raw output marked invalid: {parsed.get('reason')}")
        nodes = []
        for item in parsed.get("nodes", []):
            node_id = str(item.get("id", ""))
            if node_id not in allowed:
                parse_info["rejected_nodes"].append({"id": node_id, "reason": "not_in_per_frame_allowed_nodes"})
                continue
            nodes.append(
                Node(
                    id=node_id,
                    category=str(item.get("category", "unknown")),
                    entity_type=str(item.get("entity_type", "object" if node_id != "gripper" else "gripper")),
                    present=bool(item.get("present", True)),
                    visible=bool(item.get("visible", True)),
                    visible_pixels=int(item.get("visible_pixels", 0) or 0),
                    centroid_xy=None,
                )
            )
        node_ids = {node.id for node in nodes}
        edges = []
        for item in parsed.get("binary_edges", []):
            edge = Edge(str(item.get("subject")), str(item.get("predicate")), str(item.get("object")))
            if edge.subject not in allowed or edge.object not in allowed:
                parse_info["rejected_edges"].append({**edge.to_json(), "reason": "endpoint_not_in_per_frame_allowed_nodes"})
                continue
            if edge.predicate not in ALLOWED_PREDICATES:
                parse_info["rejected_edges"].append({**edge.to_json(), "reason": "predicate_not_allowed"})
                continue
            if edge.subject not in node_ids or edge.object not in node_ids:
                parse_info["rejected_edges"].append({**edge.to_json(), "reason": "endpoint_not_declared_in_qwen_nodes"})
                continue
            edges.append(edge)
        graph = make_graph(
            source="qwen_zero_shot",
            mode="vision_only",
            task_id=task_id,
            demo_id=demo_id,
            frame_id=frame_id,
            nodes=nodes,
            edges=edges,
            config_hash=config_hash,
            extra_metadata={"parser": "qwen_parser_v0"},
        )
        validate_graph(graph)
        return graph, parse_info
    except Exception as exc:
        parse_info["invalid"] = True
        parse_info["errors"].append(f"{type(exc).__name__}: {exc}")
        return None, parse_info
