from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "0.4"
RELATION_RULE_VERSION = "0.7"
ALLOWED_PREDICATES = (
    "left_of",
    "right_of",
    "above",
    "below",
    "front_of",
    "behind",
    "on",
    "inside",
    "contains",
    "grasping",
)
FORBIDDEN_PREDICATES = {"between", "touching", "near", "overlapping", "holding"}
ALLOWED_SOURCES = {"rule_based", "qwen_zero_shot"}
ALLOWED_MODES = {"world", "observable", "vision_only"}
ALLOWED_ENTITY_TYPES = {"object", "fixture", "gripper"}


@dataclass(frozen=True)
class Node:
    id: str
    category: str
    entity_type: str
    present: bool
    visible: bool
    visible_pixels: int = 0
    centroid_xy: Optional[Tuple[float, float]] = None

    def to_json(self, include_geometry: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "category": self.category,
            "entity_type": self.entity_type,
            "present": bool(self.present),
        }
        if include_geometry:
            payload["visible_pixels"] = int(self.visible_pixels)
            payload["centroid_xy"] = list(self.centroid_xy) if self.centroid_xy is not None else None
        return payload


@dataclass(frozen=True)
class Edge:
    subject: str
    predicate: str
    object: str

    def to_json(self) -> Dict[str, str]:
        return {"subject": self.subject, "predicate": self.predicate, "object": self.object}


def node_from_mapping(item: Dict[str, Any]) -> Node:
    centroid = item.get("centroid_xy")
    centroid_xy = None if centroid is None else (float(centroid[0]), float(centroid[1]))
    return Node(
        id=str(item["id"]),
        category=str(item.get("category", "unknown")),
        entity_type=str(item.get("entity_type", item.get("type", "object"))),
        present=bool(item.get("present", True)),
        visible=bool(item.get("visible", False)),
        visible_pixels=int(item.get("visible_pixels", 0) or 0),
        centroid_xy=centroid_xy,
    )


def edge_from_mapping(item: Dict[str, Any]) -> Edge:
    return Edge(subject=str(item["subject"]), predicate=str(item["predicate"]), object=str(item["object"]))


def validate_nodes(nodes: Iterable[Node]) -> None:
    seen = set()
    for node in nodes:
        if not node.id:
            raise ValueError("node id must be non-empty")
        if node.id in seen:
            raise ValueError(f"duplicate node id: {node.id}")
        seen.add(node.id)
        if node.entity_type not in ALLOWED_ENTITY_TYPES:
            raise ValueError(f"invalid entity_type for {node.id}: {node.entity_type}")


def validate_edges(edges: Iterable[Edge], node_ids: Iterable[str]) -> None:
    node_id_set = set(node_ids)
    for edge in edges:
        if edge.predicate not in ALLOWED_PREDICATES:
            raise ValueError(f"invalid predicate: {edge.predicate}")
        if edge.predicate in FORBIDDEN_PREDICATES:
            raise ValueError(f"forbidden predicate: {edge.predicate}")
        if edge.subject not in node_id_set:
            raise ValueError(f"edge subject is not a node: {edge.subject}")
        if edge.object not in node_id_set:
            raise ValueError(f"edge object is not a node: {edge.object}")
        if edge.predicate == "grasping" and edge.subject != "gripper":
            raise ValueError("grasping must be directed from gripper to object")


def validate_graph(graph: Dict[str, Any]) -> None:
    if graph.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if graph.get("source") not in ALLOWED_SOURCES:
        raise ValueError(f"invalid source: {graph.get('source')}")
    if graph.get("mode") not in ALLOWED_MODES:
        raise ValueError(f"invalid mode: {graph.get('mode')}")
    if "ternary_edges" in graph:
        raise ValueError("ternary_edges are forbidden")
    nodes = [node_from_mapping(item) for item in graph.get("nodes", [])]
    edges = [edge_from_mapping(item) for item in graph.get("binary_edges", [])]
    validate_nodes(nodes)
    validate_edges(edges, [node.id for node in nodes])
