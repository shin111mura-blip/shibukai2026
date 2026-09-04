#!/usr/bin/env python3
"""Utilities for oracle LIBERO scene graph probing.

This module intentionally uses simulator state only. It does not use external
detectors, segmenters, VLMs, depth estimators, or image-recognition models.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


GRAPH_VERSION = "oracle_libero_v0"
IGNORE_OBJECT_PREFIXES = (
    "floor",
    "main_table",
    "mount",
    "robot",
    "table",
    "wall",
    "world",
)
ROBOT_NAME_HINTS = ("robot", "panda", "gripper", "finger", "hand", "eef", "wrist")
CONTAINER_CATEGORIES = {"bowl", "tray", "basket", "box", "container", "drawer", "cabinet", "plate", "caddy"}
CATEGORY_RULES = (
    ("alphabet_soup", "can"),
    ("cream_cheese", "box"),
    ("chocolate_pudding", "cup"),
    ("tomato_sauce", "can"),
    ("moka_pot", "pot"),
    ("frying_pan", "pan"),
    ("wine_bottle", "bottle"),
    ("bowl", "bowl"),
    ("plate", "plate"),
    ("ramekin", "ramekin"),
    ("block", "block"),
    ("drawer", "drawer"),
    ("cabinet", "cabinet"),
    ("basket", "basket"),
    ("tray", "tray"),
    ("box", "box"),
    ("container", "container"),
    ("mug", "mug"),
    ("book", "book"),
    ("caddy", "caddy"),
    ("stove", "stove"),
    ("pot", "pot"),
    ("pan", "pan"),
    ("bottle", "bottle"),
    ("butter", "butter"),
)


@dataclass
class GraphThresholds:
    next_to: float = 0.12
    between_distance: float = 0.06
    between_min_endpoint_distance: float = 0.04
    between_min_pair_distance: float = 0.08
    on_xy: float = 0.09
    on_z: float = 0.015
    inside_xy: float = 0.10
    inside_z: float = 0.12
    grasp_distance: float = 0.08
    touching_distance: float = 0.025


def ensure_repo_paths() -> None:
    """Add local LIBERO/OpenVLA repos when scripts are run from workspace root."""
    root = Path(__file__).resolve().parents[2]
    for rel in ("openvla", "LIBERO"):
        path = str(root / rel)
        if path not in sys.path and Path(path).exists():
            sys.path.insert(0, path)


def safe_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(make_jsonable(payload), sort_keys=True) + "\n")


def make_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_jsonable(v) for v in value]
    return repr(value)


def normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def stable_node_id(raw_name: str, used: Optional[set] = None) -> str:
    base = canonical_object_name(raw_name) or "object"
    if used is None:
        return base
    if base not in used:
        used.add(base)
        return base
    idx = 1
    while f"{base}_{idx}" in used:
        idx += 1
    node_id = f"{base}_{idx}"
    used.add(node_id)
    return node_id


def infer_category(raw_name: str) -> str:
    normalized = canonical_object_name(raw_name)
    for needle, category in CATEGORY_RULES:
        if needle in normalized:
            return category
    return "unknown"


def is_robotish(name: Optional[str]) -> bool:
    normalized = normalize_name(name)
    return any(hint in normalized for hint in ROBOT_NAME_HINTS)


def canonical_object_name(name: Optional[str]) -> str:
    normalized = normalize_name(name)
    normalized = re.sub(r"_(main|root|visual|collision|col|geom|body)$", "", normalized)
    normalized = re.sub(r"_g\d+$", "", normalized)
    return normalized


def is_ignored_object_name(name: Optional[str]) -> bool:
    normalized = normalize_name(name)
    if not normalized:
        return True
    if normalized.startswith(IGNORE_OBJECT_PREFIXES):
        return True
    if normalized.endswith("_region") or "_region_" in normalized:
        return True
    if "workspace" in normalized or "target" in normalized:
        return True
    return False


def try_call(fn, *args, default=None, warnings: Optional[List[str]] = None, label: str = ""):
    try:
        return fn(*args)
    except Exception as exc:
        if warnings is not None and label:
            warnings.append(f"{label}: {type(exc).__name__}: {exc}")
        return default


def get_inner_env(env: Any) -> Any:
    return getattr(env, "env", env)


def get_sim(env: Any) -> Any:
    inner_env = get_inner_env(env)
    return getattr(inner_env, "sim", getattr(env, "sim", None))


def get_model_data(env: Any) -> Tuple[Any, Any]:
    sim = get_sim(env)
    if sim is None:
        return None, None
    return getattr(sim, "model", None), getattr(sim, "data", None)


def _id2name(model: Any, kind: str, idx: int) -> Optional[str]:
    method = getattr(model, f"{kind}_id2name", None)
    if method is None and kind == "cam":
        method = getattr(model, "camera_id2name", None)
    if method is None and kind == "joint":
        method = getattr(model, "joint_id2name", None)
    if callable(method):
        return method(idx)
    names = getattr(model, f"{kind}_names", None)
    if names is not None:
        try:
            value = names[idx]
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)
        except Exception:
            return None
    return None


def _name2id(model: Any, kind: str, name: str) -> Optional[int]:
    method = getattr(model, f"{kind}_name2id", None)
    if method is None and kind == "cam":
        method = getattr(model, "camera_name2id", None)
    if method is None and kind == "joint":
        method = getattr(model, "joint_name2id", None)
    if callable(method):
        try:
            return int(method(name))
        except Exception:
            return None
    return None


def list_model_names(model: Any, kind: str) -> List[str]:
    if model is None:
        return []
    count_attr = {"joint": "njnt"}.get(kind, f"n{kind}")
    count = int(getattr(model, count_attr, 0) or 0)
    return [name for idx in range(count) if (name := _id2name(model, kind, idx))]


def get_body_name_for_geom(model: Any, geom_name: Optional[str]) -> Optional[str]:
    if model is None or geom_name is None:
        return None
    geom_id = _name2id(model, "geom", geom_name)
    if geom_id is None:
        return None
    try:
        body_id = int(model.geom_bodyid[geom_id])
        return _id2name(model, "body", body_id)
    except Exception:
        return None


def get_position(data: Any, model: Any, kind: str, name: str) -> Optional[List[float]]:
    idx = _name2id(model, kind, name)
    if idx is None:
        return None
    attr = {"body": "body_xpos", "geom": "geom_xpos", "site": "site_xpos"}.get(kind)
    if attr is None or not hasattr(data, attr):
        return None
    try:
        return np.asarray(getattr(data, attr)[idx], dtype=float).tolist()
    except Exception:
        return None


def get_quat(data: Any, model: Any, kind: str, name: str) -> Optional[List[float]]:
    idx = _name2id(model, kind, name)
    if idx is None:
        return None
    if kind == "body" and hasattr(data, "body_xquat"):
        try:
            return np.asarray(data.body_xquat[idx], dtype=float).tolist()
        except Exception:
            return None
    if kind == "geom" and hasattr(data, "geom_xmat"):
        try:
            mat = np.asarray(data.geom_xmat[idx], dtype=float).reshape(3, 3)
            return rotmat_to_quat_wxyz(mat)
        except Exception:
            return None
    return None


def rotmat_to_quat_wxyz(mat: np.ndarray) -> List[float]:
    trace = float(np.trace(mat))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return [0.25 * s, (mat[2, 1] - mat[1, 2]) / s, (mat[0, 2] - mat[2, 0]) / s, (mat[1, 0] - mat[0, 1]) / s]
    axis = int(np.argmax([mat[0, 0], mat[1, 1], mat[2, 2]]))
    if axis == 0:
        s = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
        return [(mat[2, 1] - mat[1, 2]) / s, 0.25 * s, (mat[0, 1] + mat[1, 0]) / s, (mat[0, 2] + mat[2, 0]) / s]
    if axis == 1:
        s = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
        return [(mat[0, 2] - mat[2, 0]) / s, (mat[0, 1] + mat[1, 0]) / s, 0.25 * s, (mat[1, 2] + mat[2, 1]) / s]
    s = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
    return [(mat[1, 0] - mat[0, 1]) / s, (mat[0, 2] + mat[2, 0]) / s, (mat[1, 2] + mat[2, 1]) / s, 0.25 * s]


def collect_candidate_object_names(env: Any, model: Any) -> Dict[str, List[str]]:
    inner_env = get_inner_env(env)
    sources: Dict[str, List[str]] = {}

    for attr in ("objects_dict", "object_states_dict", "object_cfgs"):
        value = getattr(inner_env, attr, None)
        if isinstance(value, dict):
            sources[attr] = sorted(str(key) for key in value.keys())

    for attr in ("objects", "fixtures"):
        values = getattr(inner_env, attr, None)
        names = []
        if values:
            if isinstance(values, dict):
                values = values.values()
            for obj in values:
                name = getattr(obj, "name", None)
                if name:
                    names.append(str(name))
        if names:
            sources[attr] = sorted(set(names))

    for attr in ("obj_of_interest",):
        values = getattr(inner_env, attr, None)
        if values:
            sources[attr] = sorted(set(str(v) for v in values))

    body_names = list_model_names(model, "body")
    geom_names = list_model_names(model, "geom")
    sources["mujoco_bodies_heuristic"] = [
        name for name in body_names if not is_ignored_object_name(name) and not is_robotish(name)
    ]
    sources["mujoco_geoms_heuristic"] = [
        name for name in geom_names if not is_ignored_object_name(name) and not is_robotish(name)
    ]
    return sources


def extract_object_nodes(env: Any, warnings: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], Dict[str, str]]:
    model, data = get_model_data(env)
    if model is None or data is None:
        return [], {}, {}

    sources = collect_candidate_object_names(env, model)
    raw_names: List[str] = []
    for source in ("objects_dict", "object_states_dict", "objects", "obj_of_interest", "fixtures", "mujoco_bodies_heuristic"):
        raw_names.extend(sources.get(source, []))

    body_names = list_model_names(model, "body")
    geom_names = list_model_names(model, "geom")
    used_ids: set = set()
    nodes: List[Dict[str, Any]] = []
    object_geoms: Dict[str, List[str]] = {}
    geom_to_node: Dict[str, str] = {}
    seen_norms: set = set()

    for raw_name in raw_names:
        norm = canonical_object_name(raw_name)
        if not norm or norm in seen_norms or is_ignored_object_name(norm) or is_robotish(norm):
            continue
        seen_norms.add(norm)
        matching_bodies = [b for b in body_names if canonical_object_name(b) == norm or normalize_name(b).startswith(f"{norm}_")]
        body_name = raw_name if raw_name in body_names else (matching_bodies[0] if matching_bodies else None)
        matching_geoms = [
            geom
            for geom in geom_names
            if canonical_object_name(geom) == norm or normalize_name(geom).startswith(f"{norm}_")
        ]
        if not body_name and matching_geoms:
            body_name = get_body_name_for_geom(model, matching_geoms[0])

        pos = get_position(data, model, "body", body_name) if body_name else None
        quat = get_quat(data, model, "body", body_name) if body_name else None
        if pos is None and matching_geoms:
            pos = get_position(data, model, "geom", matching_geoms[0])
            quat = get_quat(data, model, "geom", matching_geoms[0])
        if pos is None:
            continue

        node_id = stable_node_id(raw_name, used_ids)
        object_geoms[node_id] = sorted(set(matching_geoms))
        for geom in matching_geoms:
            geom_to_node[geom] = node_id
        nodes.append(
            {
                "id": node_id,
                "type": "object",
                "category": infer_category(raw_name),
                "name_raw": raw_name,
                "name_normalized": norm,
                "body_name": body_name,
                "geom_names": sorted(set(matching_geoms)),
                "pos_world": pos,
                "quat_world": quat,
                "bbox2d": None,
                "visible": None,
            }
        )

    gripper_node = extract_gripper_node(env)
    if gripper_node is not None:
        nodes.append(gripper_node)
    if warnings is not None and not nodes:
        warnings.append("No object or gripper nodes were extracted from simulator state.")
    return nodes, object_geoms, geom_to_node


def extract_gripper_node(env: Any) -> Optional[Dict[str, Any]]:
    model, data = get_model_data(env)
    if model is None or data is None:
        return None
    site_names = list_model_names(model, "site")
    geom_names = list_model_names(model, "geom")
    body_names = list_model_names(model, "body")
    site_candidates = [s for s in site_names if "grip" in normalize_name(s) or "eef" in normalize_name(s)]
    geom_candidates = [g for g in geom_names if is_robotish(g) and ("finger" in normalize_name(g) or "gripper" in normalize_name(g))]
    body_candidates = [b for b in body_names if is_robotish(b) and ("hand" in normalize_name(b) or "gripper" in normalize_name(b))]
    pos = None
    quat = None
    source = None
    for site in site_candidates:
        pos = get_position(data, model, "site", site)
        if pos is not None:
            source = {"kind": "site", "name": site}
            break
    if pos is None:
        points = [get_position(data, model, "geom", geom) for geom in geom_candidates]
        points = [np.asarray(p, dtype=float) for p in points if p is not None]
        if points:
            pos = np.mean(np.stack(points, axis=0), axis=0).tolist()
            source = {"kind": "geom_mean", "names": geom_candidates}
    if pos is None:
        for body in body_candidates:
            pos = get_position(data, model, "body", body)
            quat = get_quat(data, model, "body", body)
            if pos is not None:
                source = {"kind": "body", "name": body}
                break
    if pos is None:
        return None
    return {
        "id": "gripper",
        "type": "robot",
        "category": "gripper",
        "name_raw": "gripper",
        "body_name": source.get("name") if isinstance(source, dict) and source.get("kind") == "body" else None,
        "geom_names": sorted(set(geom_candidates)),
        "site_names": sorted(set(site_candidates)),
        "pos_world": pos,
        "quat_world": quat,
        "bbox2d": None,
        "visible": None,
        "source": source,
    }


def get_contacts(env: Any, geom_to_node: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    model, data = get_model_data(env)
    if model is None or data is None or not hasattr(data, "ncon") or not hasattr(data, "contact"):
        return []
    contacts: List[Dict[str, Any]] = []
    for idx in range(int(data.ncon)):
        try:
            contact = data.contact[idx]
            geom1 = _id2name(model, "geom", int(contact.geom1))
            geom2 = _id2name(model, "geom", int(contact.geom2))
        except Exception:
            continue
        body1 = get_body_name_for_geom(model, geom1)
        body2 = get_body_name_for_geom(model, geom2)
        contacts.append(
            {
                "contact_index": idx,
                "geom1": geom1,
                "geom2": geom2,
                "body1": body1,
                "body2": body2,
                "node1": geom_to_node.get(geom1) if geom_to_node else None,
                "node2": geom_to_node.get(geom2) if geom_to_node else None,
            }
        )
    return contacts


def build_edges(nodes: List[Dict[str, Any]], contacts: List[Dict[str, Any]], thresholds: GraphThresholds) -> List[Dict[str, Any]]:
    object_nodes = [n for n in nodes if n.get("type") == "object" and n.get("pos_world") is not None]
    gripper = next((n for n in nodes if n.get("id") == "gripper" and n.get("pos_world") is not None), None)
    edges: List[Dict[str, Any]] = []

    for i, src in enumerate(object_nodes):
        for dst in object_nodes[i + 1 :]:
            distance = xy_distance(src["pos_world"], dst["pos_world"])
            if distance < thresholds.next_to:
                debug = {"distance_xy": distance, "threshold": thresholds.next_to}
                edges.append(edge(src["id"], "next_to", dst["id"], debug))
                edges.append(edge(dst["id"], "next_to", src["id"], debug))

    for a in object_nodes:
        for i, b in enumerate(object_nodes):
            if b["id"] == a["id"]:
                continue
            for c in object_nodes[i + 1 :]:
                if c["id"] in {a["id"], b["id"]}:
                    continue
                debug = between_debug(a["pos_world"], b["pos_world"], c["pos_world"])
                if (
                    debug["pair_distance_xy"] >= thresholds.between_min_pair_distance
                    and debug["distance_to_segment_xy"] <= thresholds.between_distance
                    and 0.0 <= debug["projection_ratio"] <= 1.0
                    and min(debug["endpoint_distance_xy"]) >= thresholds.between_min_endpoint_distance
                ):
                    debug.update(
                        {
                            "threshold": thresholds.between_distance,
                            "min_endpoint_threshold": thresholds.between_min_endpoint_distance,
                            "min_pair_distance_threshold": thresholds.between_min_pair_distance,
                        }
                    )
                    edges.append(edge(a["id"], "between", [b["id"], c["id"]], debug))

    contact_pairs = node_contact_pairs(contacts)
    for src, dst, contact_list in contact_pairs:
        debug = {"contacts": contact_list}
        edges.append(edge(src, "touching", dst, debug, source="mujoco_contact"))
        edges.append(edge(dst, "touching", src, debug, source="mujoco_contact"))

    for a in object_nodes:
        for b in object_nodes:
            if a["id"] == b["id"]:
                continue
            dist = xy_distance(a["pos_world"], b["pos_world"])
            dz = float(a["pos_world"][2] - b["pos_world"][2])
            touching = frozenset((a["id"], b["id"])) in {frozenset((p[0], p[1])) for p in contact_pairs}
            if dz > thresholds.on_z and dist < thresholds.on_xy and touching:
                edges.append(edge(a["id"], "on", b["id"], {"distance_xy": dist, "delta_z": dz, "requires_contact": True, "threshold_xy": thresholds.on_xy, "threshold_z": thresholds.on_z}))
            if b.get("category") in CONTAINER_CATEGORIES and dist < thresholds.inside_xy and abs(dz) < thresholds.inside_z:
                edges.append(edge(a["id"], "inside", b["id"], {"distance_xy": dist, "delta_z": dz, "container_category": b.get("category"), "threshold_xy": thresholds.inside_xy, "threshold_z_abs": thresholds.inside_z}))

    if gripper is not None:
        gripper_touching = set()
        for contact in contacts:
            n1, n2 = contact.get("node1"), contact.get("node2")
            g1, g2 = contact.get("geom1"), contact.get("geom2")
            if n1 and is_robotish(g2):
                gripper_touching.add(n1)
            if n2 and is_robotish(g1):
                gripper_touching.add(n2)
        for obj in object_nodes:
            distance = euclidean_distance(gripper["pos_world"], obj["pos_world"])
            if obj["id"] in gripper_touching:
                edges.append(edge("gripper", "touching", obj["id"], {"distance": distance, "source_condition": "robot_geom_contact"}, source="mujoco_contact"))
            if obj["id"] in gripper_touching and distance < thresholds.grasp_distance:
                edges.append(
                    edge(
                        "gripper",
                        "grasping",
                        obj["id"],
                        {"distance": distance, "threshold": thresholds.grasp_distance, "conditions": ["gripper_touching_object", "object_close_to_gripper"]},
                    )
                )
    return dedupe_edges(edges)


def node_contact_pairs(contacts: List[Dict[str, Any]]) -> List[Tuple[str, str, List[Dict[str, str]]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for contact in contacts:
        n1, n2 = contact.get("node1"), contact.get("node2")
        if not n1 or not n2 or n1 == n2:
            continue
        key = tuple(sorted((n1, n2)))
        grouped.setdefault(key, []).append(
            {
                "geom1": contact.get("geom1"),
                "geom2": contact.get("geom2"),
                "body1": contact.get("body1"),
                "body2": contact.get("body2"),
            }
        )
    return [(src, dst, pairs) for (src, dst), pairs in grouped.items()]


def edge(src: str, rel: str, dst: Any, debug: Dict[str, Any], confidence: float = 1.0, source: str = "oracle_rule") -> Dict[str, Any]:
    return {"src": src, "rel": rel, "dst": dst, "confidence": confidence, "source": source, "rule_debug": debug}


def dedupe_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped = []
    seen = set()
    for item in edges:
        key = json.dumps({"src": item.get("src"), "rel": item.get("rel"), "dst": item.get("dst")}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a[:2], dtype=float) - np.asarray(b[:2], dtype=float)))


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def between_debug(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> Dict[str, Any]:
    pa = np.asarray(a[:2], dtype=float)
    pb = np.asarray(b[:2], dtype=float)
    pc = np.asarray(c[:2], dtype=float)
    vec = pc - pb
    denom = float(np.dot(vec, vec))
    if denom <= 1e-12:
        ratio = 0.0
        closest = pb
    else:
        ratio = float(np.dot(pa - pb, vec) / denom)
        closest = pb + ratio * vec
    return {
        "distance_to_segment_xy": float(np.linalg.norm(pa - closest)),
        "projection_ratio": ratio,
        "endpoint_distance_xy": [float(np.linalg.norm(pa - pb)), float(np.linalg.norm(pa - pc))],
        "pair_distance_xy": float(np.linalg.norm(pc - pb)),
    }


def make_graph_record(
    suite: str,
    task_id: int,
    task_name: str,
    instruction: str,
    episode_id: int,
    timestep: int,
    env: Any,
    thresholds: GraphThresholds,
    warnings: Optional[List[str]] = None,
    camera_name: str = "agentview",
    image_width: int = 128,
    image_height: int = 128,
) -> Dict[str, Any]:
    warnings = warnings if warnings is not None else []
    nodes, object_geoms, geom_to_node = extract_object_nodes(env, warnings=warnings)
    cam_info = camera_info(env)
    selected_camera = next((cam for cam in cam_info.get("cameras", []) if cam.get("name") == camera_name), None)
    projected_count = 0
    if selected_camera is not None:
        for node in nodes:
            if node.get("pos_world") is None:
                node["center2d"] = None
                continue
            center2d = project_world_to_image(node["pos_world"], selected_camera, image_width, image_height)
            node["center2d"] = center2d
            if center2d is not None:
                u, v = center2d
                node["visible"] = bool(0.0 <= u < image_width and 0.0 <= v < image_height)
                node["bbox2d"] = [u, v, u, v]
                projected_count += 1
            else:
                node["visible"] = False
    contacts = get_contacts(env, geom_to_node)
    edges = build_edges(nodes, contacts, thresholds)
    sim = get_sim(env)
    sim_time = None
    try:
        sim_time = float(sim.data.time)
    except Exception:
        pass
    return {
        "suite": suite,
        "task_id": int(task_id),
        "task_name": task_name,
        "episode_id": int(episode_id),
        "timestep": int(timestep),
        "sim_time": sim_time,
        "instruction": instruction,
        "nodes": nodes,
        "edges": edges,
        "contacts": contacts,
        "metadata": {
            "graph_version": GRAPH_VERSION,
            "generator": "generate_oracle_scene_graphs.py",
            "object_extraction": "LIBERO env registries plus MuJoCo body/geom heuristics",
            "relation_rules": "rule_based_oracle_from_simulator_state",
            "camera_projection": {
                "camera_name": camera_name,
                "image_width": image_width,
                "image_height": image_height,
                "available": selected_camera is not None,
                "projected_node_count": projected_count,
            },
            "warnings": warnings,
        },
    }


def import_libero_modules() -> Tuple[Any, Any, Any]:
    ensure_repo_paths()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv

    return benchmark, get_libero_path, (OffScreenRenderEnv, SegmentationRenderEnv)


def create_libero_env(
    suite: str,
    task_id: int,
    image_size: int = 128,
    camera_depths: bool = False,
    camera_segmentations: Optional[str] = None,
    seed: int = 0,
):
    benchmark, get_libero_path, env_classes = import_libero_modules()
    OffScreenRenderEnv, SegmentationRenderEnv = env_classes
    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_cls = SegmentationRenderEnv if camera_segmentations else OffScreenRenderEnv
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": image_size,
        "camera_widths": image_size,
        "camera_depths": camera_depths,
        "camera_segmentations": camera_segmentations,
    }
    env = env_cls(**env_args)
    try:
        env.seed(seed)
    except Exception:
        pass
    init_states = task_suite.get_task_init_states(task_id)
    metadata = {
        "suite": suite,
        "task_id": task_id,
        "num_tasks": getattr(task_suite, "n_tasks", None),
        "task_name": getattr(task, "name", str(task_id)),
        "instruction": getattr(task, "language", ""),
        "bddl_file": bddl_file,
        "init_states_count": len(init_states) if hasattr(init_states, "__len__") else None,
        "benchmark_names": sorted(benchmark.get_benchmark_dict().keys()),
    }
    return env, task_suite, task, init_states, metadata


def reset_env_to_episode(env: Any, init_states: Any, episode_id: int = 0) -> Tuple[Any, List[str]]:
    warnings: List[str] = []
    obs = None
    try:
        env.reset()
    except Exception as exc:
        warnings.append(f"env.reset failed: {type(exc).__name__}: {exc}")
        raise
    if init_states is not None and hasattr(init_states, "__len__") and len(init_states) > 0:
        idx = int(episode_id) % len(init_states)
        try:
            obs = env.set_init_state(init_states[idx])
        except Exception as exc:
            warnings.append(f"env.set_init_state failed for index {idx}: {type(exc).__name__}: {exc}")
            raise
    return obs, warnings


def observation_summary(obs: Any) -> Dict[str, Any]:
    if not isinstance(obs, dict):
        return {"type": type(obs).__name__, "keys": []}
    summary = {"type": "dict", "keys": sorted(obs.keys()), "items": {}}
    for key, value in obs.items():
        item = {"type": type(value).__name__}
        if hasattr(value, "shape"):
            item["shape"] = list(value.shape)
            item["dtype"] = str(getattr(value, "dtype", ""))
        elif isinstance(value, (list, tuple)):
            item["length"] = len(value)
        summary["items"][key] = item
    return summary


def camera_info(env: Any, obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    model, data = get_model_data(env)
    info: Dict[str, Any] = {
        "available": model is not None and data is not None,
        "cameras": [],
        "rgb_observation_keys": [],
        "depth_observation_keys": [],
        "segmentation_observation_keys": [],
        "projection_available": False,
        "notes": [],
    }
    if isinstance(obs, dict):
        for key, value in obs.items():
            if "image" in key and hasattr(value, "shape"):
                info["rgb_observation_keys"].append(key)
            if "depth" in key and hasattr(value, "shape"):
                info["depth_observation_keys"].append(key)
            if "segmentation" in key and hasattr(value, "shape"):
                info["segmentation_observation_keys"].append(key)
    if model is None or data is None:
        return info
    for cam_name in list_model_names(model, "cam"):
        cam_id = _name2id(model, "cam", cam_name)
        entry = {"name": cam_name, "id": cam_id, "pos_world": None, "xmat": None, "fovy": None, "intrinsics": None, "extrinsics": None}
        if cam_id is not None:
            try:
                entry["pos_world"] = np.asarray(data.cam_xpos[cam_id], dtype=float).tolist()
                entry["xmat"] = np.asarray(data.cam_xmat[cam_id], dtype=float).reshape(3, 3).tolist()
            except Exception as exc:
                info["notes"].append(f"camera pose unavailable for {cam_name}: {exc}")
            try:
                entry["fovy"] = float(model.cam_fovy[cam_id])
            except Exception:
                pass
        info["cameras"].append(entry)
    info["projection_available"] = bool(info["cameras"])
    return info


def project_world_to_image(point_world: Sequence[float], camera: Dict[str, Any], width: int, height: int) -> Optional[List[float]]:
    if not camera.get("pos_world") or not camera.get("xmat") or camera.get("fovy") is None:
        return None
    pos = np.asarray(camera["pos_world"], dtype=float)
    rot = np.asarray(camera["xmat"], dtype=float).reshape(3, 3)
    point = np.asarray(point_world, dtype=float)
    cam = rot.T @ (point - pos)
    if cam[2] >= -1e-6:
        return None
    fovy = math.radians(float(camera["fovy"]))
    fy = 0.5 * height / math.tan(fovy / 2.0)
    fx = fy
    u = width / 2.0 + fx * (cam[0] / -cam[2])
    v = height / 2.0 - fy * (cam[1] / -cam[2])
    return [float(u), float(v)]


def save_rgb_from_obs(obs: Any, output_path: Path) -> Optional[Path]:
    if not isinstance(obs, dict):
        return None
    key = "agentview_image" if "agentview_image" in obs else next((k for k in obs if "image" in k and hasattr(obs[k], "shape")), None)
    if key is None:
        return None
    try:
        from PIL import Image

        image = np.asarray(obs[key])
        if image.ndim == 3 and image.shape[-1] >= 3:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image[..., :3].astype(np.uint8)).save(output_path)
            return output_path
    except Exception:
        return None
    return None


def environment_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {"python": sys.version, "python_executable": sys.executable}
    for module_name in ("libero", "robosuite", "mujoco", "mujoco_py", "dm_control"):
        try:
            module = __import__(module_name)
            versions[module_name] = {"available": True, "version": getattr(module, "__version__", None), "file": getattr(module, "__file__", None)}
        except Exception as exc:
            versions[module_name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return versions


def exception_payload(exc: BaseException) -> Dict[str, Any]:
    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
