from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ALLOWED_PREDICATES: Tuple[str, ...] = (
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


def read_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def json_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_graph(graph: Dict[str, Any], *, allowed_predicates: Sequence[str] = ALLOWED_PREDICATES) -> List[str]:
    errors: List[str] = []
    allowed = set(allowed_predicates)
    node_ids = set()
    for idx, node in enumerate(graph.get("nodes", [])):
        for key in ("id", "category", "entity_type", "present"):
            if key not in node:
                errors.append(f"node[{idx}] missing {key}")
        node_id = node.get("id")
        if node_id in node_ids:
            errors.append(f"duplicate node id {node_id}")
        node_ids.add(node_id)
        if node.get("entity_type") not in {"object", "fixture", "gripper"}:
            errors.append(f"node {node_id} invalid entity_type {node.get('entity_type')}")
        if node.get("present") is not True:
            errors.append(f"node {node_id} present is not true")
    for idx, edge in enumerate(graph.get("binary_edges", [])):
        for key in ("subject", "predicate", "object"):
            if key not in edge:
                errors.append(f"edge[{idx}] missing {key}")
        s = edge.get("subject")
        o = edge.get("object")
        p = edge.get("predicate")
        if s not in node_ids:
            errors.append(f"edge[{idx}] subject {s} not in nodes")
        if o not in node_ids:
            errors.append(f"edge[{idx}] object {o} not in nodes")
        if s == o:
            errors.append(f"edge[{idx}] self edge {s}")
        if p not in allowed:
            errors.append(f"edge[{idx}] predicate {p} not allowed")
    return errors


def graph_node_ids(graph: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(sorted(node["id"] for node in graph.get("nodes", [])))


def graph_triplets(graph: Dict[str, Any]) -> Tuple[Tuple[str, str, str], ...]:
    return tuple(
        sorted((edge["subject"], edge["predicate"], edge["object"]) for edge in graph.get("binary_edges", []))
    )


def canonical_graph_key(graph: Dict[str, Any]) -> str:
    canonical = {
        "nodes": graph_node_ids(graph),
        "binary_edges": graph_triplets(graph),
    }
    return json_sha256(canonical)


def compact_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    nodes = sorted(
        (
            {
                "id": n["id"],
                "category": n["category"],
                "entity_type": n["entity_type"],
                "present": True,
            }
            for n in graph.get("nodes", [])
        ),
        key=lambda x: x["id"],
    )
    edges = sorted(
        (
            {"subject": e["subject"], "predicate": e["predicate"], "object": e["object"]}
            for e in graph.get("binary_edges", [])
        ),
        key=lambda x: (x["subject"], x["predicate"], x["object"]),
    )
    return {"nodes": nodes, "binary_edges": edges}


def iter_graph_paths(graph_root: Path) -> Iterable[Path]:
    yield from sorted(graph_root.glob("task_*/global_*/*.json"))


def parse_graph_path(path: Path) -> Tuple[int, int, int]:
    task = int(path.parents[1].name.split("_")[1])
    episode = int(path.parent.name.split("_")[1])
    frame = int(path.stem)
    return task, episode, frame

