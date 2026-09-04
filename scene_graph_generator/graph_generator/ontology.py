from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .schema import ALLOWED_PREDICATES, iter_graph_paths, json_sha256, read_json, write_json


def build_ontology(graph_root: Path) -> Dict[str, Any]:
    node_meta: Dict[str, Dict[str, str]] = {}
    conflicts = []
    for path in iter_graph_paths(graph_root):
        graph = read_json(path)
        for node in graph.get("nodes", []):
            node_id = node["id"]
            meta = {"category": node["category"], "entity_type": node["entity_type"]}
            if node_id in node_meta and node_meta[node_id] != meta:
                conflicts.append({"node_id": node_id, "first": node_meta[node_id], "new": meta, "path": str(path)})
            node_meta.setdefault(node_id, meta)
    if conflicts:
        raise ValueError(f"Ontology conflicts found: {conflicts[:5]}")
    nodes = {
        node_id: {"index": idx, **node_meta[node_id]}
        for idx, node_id in enumerate(sorted(node_meta))
    }
    predicates = {pred: idx for idx, pred in enumerate(ALLOWED_PREDICATES)}
    ontology = {"nodes": nodes, "predicates": predicates}
    ontology["ontology_hash"] = json_sha256(ontology)
    return ontology


def save_ontology(ontology: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "ontology.json", ontology)
    (output_dir / "ontology_hash.txt").write_text(ontology["ontology_hash"] + "\n")


def load_ontology(path: Path) -> Dict[str, Any]:
    return read_json(path)

