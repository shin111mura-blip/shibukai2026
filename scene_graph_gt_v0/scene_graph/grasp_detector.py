from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Set

from .schema import Edge


@dataclass(frozen=True)
class FingerGeomSets:
    left: tuple[str, ...]
    right: tuple[str, ...]


def infer_finger_geoms(geom_names: Sequence[str]) -> FingerGeomSets:
    left = []
    right = []
    for name in geom_names:
        lowered = name.lower()
        if ("left" in lowered or "finger1" in lowered) and ("pad" in lowered or "finger" in lowered):
            left.append(name)
        if ("right" in lowered or "finger2" in lowered) and ("pad" in lowered or "finger" in lowered):
            right.append(name)
    if not left:
        left = [name for name in geom_names if "l_" in name.lower() and "finger" in name.lower()]
    if not right:
        right = [name for name in geom_names if "r_" in name.lower() and "finger" in name.lower()]
    return FingerGeomSets(tuple(sorted(set(left))), tuple(sorted(set(right))))


def contact_geom_names(contacts: Iterable[Dict[str, Any]], finger_geoms: Sequence[str], object_geoms: Sequence[str]) -> Set[str]:
    fingers = set(finger_geoms)
    objects = set(object_geoms)
    matched: Set[str] = set()
    for contact in contacts:
        g1 = contact.get("geom1")
        g2 = contact.get("geom2")
        if g1 in fingers and g2 in objects:
            matched.add(str(g1))
        if g2 in fingers and g1 in objects:
            matched.add(str(g2))
    return matched


def detect_grasping_for_object(
    *,
    contacts: Iterable[Dict[str, Any]],
    object_id: str,
    object_geoms: Sequence[str],
    finger_geoms: FingerGeomSets,
    official_grasp_result: bool | None = None,
) -> tuple[bool, Dict[str, Any]]:
    contacts = list(contacts)
    left_contacts = contact_geom_names(contacts, finger_geoms.left, object_geoms)
    right_contacts = contact_geom_names(contacts, finger_geoms.right, object_geoms)
    contact_grasp = bool(left_contacts and right_contacts)
    if official_grasp_result is None:
        result = contact_grasp
        rule = "fallback_left_and_right_finger_contact"
    else:
        result = bool(official_grasp_result)
        rule = "libero_official_check_grasp"
    diagnostics = {
        "left_finger_contact": bool(left_contacts),
        "right_finger_contact": bool(right_contacts),
        "contact_grasp_result": contact_grasp,
        "official_grasp_result": bool(official_grasp_result) if official_grasp_result is not None else None,
        "rule": rule,
        "left_finger_geoms": sorted(left_contacts),
        "right_finger_geoms": sorted(right_contacts),
        "object_contact_geoms": sorted(set(object_geoms)),
    }
    return result, diagnostics


def grasping_edges(
    contacts: Iterable[Dict[str, Any]],
    object_geom_map: Dict[str, Sequence[str]],
    finger_geoms: FingerGeomSets,
    official_results: Dict[str, bool] | None = None,
) -> tuple[List[Edge], Dict[str, Any]]:
    edges: List[Edge] = []
    diagnostics: Dict[str, Any] = {}
    for object_id in sorted(object_geom_map):
        result, diag = detect_grasping_for_object(
            contacts=contacts,
            object_id=object_id,
            object_geoms=object_geom_map[object_id],
            finger_geoms=finger_geoms,
            official_grasp_result=(official_results or {}).get(object_id),
        )
        diagnostics[object_id] = diag
        if result:
            edges.append(Edge("gripper", "grasping", object_id))
    return edges, diagnostics
