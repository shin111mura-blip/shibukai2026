from __future__ import annotations

import os
import re
import sys
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schema import Node


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_repo_paths() -> None:
    root = repo_root()
    for rel in ("LIBERO", "openvla"):
        path = str(root / rel)
        if Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)


def normalize_name(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def parse_bddl_problem(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    language = ""
    match = re.search(r"\(:language\s+(.+?)\)", text)
    if match:
        language = match.group(1).strip()

    def typed_block(block_name: str) -> Dict[str, str]:
        block_match = re.search(rf"\(:{block_name}\s+(.*?)\n\s*\)", text, flags=re.S)
        result: Dict[str, str] = {}
        if not block_match:
            return result
        for line in block_match.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith(";") or " - " not in line:
                continue
            left, category = line.split(" - ", 1)
            for item in left.split():
                result[item.strip()] = category.strip()
        return result

    obj_interest_match = re.search(r"\(:obj_of_interest\s+(.*?)\n\s*\)", text, flags=re.S)
    obj_of_interest = obj_interest_match.group(1).split() if obj_interest_match else []
    return {
        "language": language,
        "objects": typed_block("objects"),
        "fixtures": typed_block("fixtures"),
        "obj_of_interest": obj_of_interest,
        "text": text,
    }


def candidate_nodes_from_bddl(path: Path, include_fixtures: bool = True) -> List[Node]:
    parsed = parse_bddl_problem(path)
    nodes = [
        Node(id=name, category=category, entity_type="object", present=True, visible=False)
        for name, category in sorted(parsed["objects"].items())
    ]
    if include_fixtures:
        for name, category in sorted(parsed["fixtures"].items()):
            if normalize_name(name) in {"main_table"}:
                continue
            nodes.append(Node(id=name, category=category, entity_type="fixture", present=True, visible=False))
    nodes.append(Node(id="gripper", category="gripper", entity_type="gripper", present=True, visible=False))
    return nodes


def segmentation_key(obs: Dict[str, Any], camera_name: str, kind: str) -> Optional[str]:
    preferred = f"{camera_name}_segmentation_{kind}"
    if preferred in obs:
        return preferred
    return next((key for key in obs if camera_name in key and "segmentation" in key and kind in key), None)


def rgb_key(obs: Dict[str, Any], camera_name: str) -> Optional[str]:
    preferred = f"{camera_name}_image"
    if preferred in obs:
        return preferred
    return next((key for key in obs if camera_name in key and "image" in key and "segmentation" not in key), None)


def segmentation_stats(mask: Any) -> Tuple[int, Optional[Tuple[float, float]]]:
    import numpy as np

    mask = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask)
    count = int(xs.size)
    if count == 0:
        return 0, None
    return count, (float(xs.mean()), float(ys.mean()))


def model_name_to_id(model: Any, kind: str, name: str) -> Optional[int]:
    method_names = [f"{kind}_name2id"]
    if kind == "camera":
        method_names.append("cam_name2id")
    for method_name in method_names:
        method = getattr(model, method_name, None)
        if callable(method):
            try:
                return int(method(name))
            except Exception:
                pass
    return None


def site_world_position(model: Any, data: Any, site_name: str) -> Optional[Any]:
    getter = getattr(data, "get_site_xpos", None)
    if callable(getter):
        try:
            return getter(site_name)
        except Exception:
            pass
    site_id = model_name_to_id(model, "site", site_name)
    if site_id is None:
        return None
    try:
        return data.site_xpos[site_id]
    except Exception:
        return None


def indexed_world_position(model: Any, data: Any, kind: str, idx: int) -> Optional[Any]:
    attr = {"body": "body_xpos", "geom": "geom_xpos", "site": "site_xpos"}.get(kind)
    if attr is None:
        return None
    try:
        return getattr(data, attr)[idx]
    except Exception:
        return None


def named_world_position(model: Any, data: Any, kind: str, name: str) -> Optional[Any]:
    idx = model_name_to_id(model, kind, name)
    if idx is None:
        return None
    return indexed_world_position(model, data, kind, idx)


def prefixed_world_position(model: Any, data: Any, kind: str, node_id: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    import numpy as np

    target = normalize_name(node_id)
    names = model_names(model, kind)
    exact = [name for name in names if normalize_name(name) == target]
    prefixed = [name for name in names if normalize_name(name).startswith(target)]
    selected = exact or prefixed
    positions = []
    for name in selected:
        pos = named_world_position(model, data, kind, name)
        if pos is not None:
            positions.append(np.asarray(pos, dtype=float))
    diagnostic = {"kind": kind, "matched_names": selected}
    if not positions:
        return None, diagnostic
    return np.mean(np.stack(positions, axis=0), axis=0), diagnostic


def node_world_position(env: Any, node_id: str, entity_type: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return None, {"error": "missing_sim_model_or_data"}
    if node_id == "gripper" or entity_type == "gripper":
        point = site_world_position(model, data, "gripper0_grip_site")
        return point, {"kind": "site", "matched_names": ["gripper0_grip_site"] if point is not None else []}

    for kind in ("body", "geom"):
        point, diagnostic = prefixed_world_position(model, data, kind, node_id)
        if point is not None:
            return point, diagnostic
    return None, {"error": "node_position_unavailable"}


def camera_record(model: Any, data: Any, camera_name: str) -> Optional[Dict[str, Any]]:
    camera_id = model_name_to_id(model, "camera", camera_name)
    if camera_id is None:
        return None
    try:
        return {
            "id": camera_id,
            "pos_world": data.cam_xpos[camera_id],
            "xmat": data.cam_xmat[camera_id],
            "fovy": float(model.cam_fovy[camera_id]),
        }
    except Exception:
        return None


def project_world_to_image(point_world: Any, camera: Dict[str, Any], width: int, height: int) -> Optional[Tuple[float, float]]:
    import numpy as np

    try:
        pos = np.asarray(camera["pos_world"], dtype=float)
        rot = np.asarray(camera["xmat"], dtype=float).reshape(3, 3)
        point = np.asarray(point_world, dtype=float)
        cam = rot.T @ (point - pos)
        if cam[2] >= -1e-6:
            return None
        fovy = math.radians(float(camera["fovy"]))
        fy = 0.5 * height / math.tan(fovy / 2.0)
        fx = fy
        x = width / 2.0 + fx * (cam[0] / -cam[2])
        y = height / 2.0 - fy * (cam[1] / -cam[2])
        return float(x), float(y)
    except Exception:
        return None


def gripper_site_centroid(
    *,
    env: Any,
    camera_name: str,
    image_width: int,
    image_height: int,
    site_name: str = "gripper0_grip_site",
) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    diagnostic: Dict[str, Any] = {"method": "grip_site_projection", "site_name": site_name, "camera_name": camera_name}
    if model is None or data is None:
        diagnostic["error"] = "missing_sim_model_or_data"
        return None, diagnostic

    camera = camera_record(model, data, camera_name)
    if camera is None:
        diagnostic["error"] = "camera_pose_unavailable"
        return None, diagnostic
    diagnostic["camera_id"] = camera["id"]

    point_world = site_world_position(model, data, site_name)
    if point_world is None:
        diagnostic["error"] = "site_position_unavailable"
        return None, diagnostic
    try:
        import numpy as np

        diagnostic["site_world_xyz"] = np.asarray(point_world, dtype=float).tolist()
    except Exception:
        pass

    centroid = project_world_to_image(point_world, camera, image_width, image_height)
    diagnostic["centroid_xy"] = list(centroid) if centroid is not None else None
    if centroid is None:
        diagnostic["error"] = "projection_failed"
    return centroid, diagnostic


def gripper_finger_pad_centroid(
    *,
    env: Any,
    camera_name: str,
    image_width: int,
    image_height: int,
) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
    import numpy as np

    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    names = ["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"]
    diagnostic: Dict[str, Any] = {"method": "finger_pad_projection", "geom_names": names, "camera_name": camera_name}
    if model is None or data is None:
        diagnostic["error"] = "missing_sim_model_or_data"
        return None, diagnostic
    camera = camera_record(model, data, camera_name)
    if camera is None:
        diagnostic["error"] = "camera_pose_unavailable"
        return None, diagnostic

    points = []
    world_points = []
    for name in names:
        point_world = named_world_position(model, data, "geom", name)
        if point_world is None:
            continue
        projected = project_world_to_image(point_world, camera, image_width, image_height)
        if projected is None:
            continue
        points.append(projected)
        world_points.append(np.asarray(point_world, dtype=float).tolist())
    diagnostic["geom_world_xyz"] = world_points
    diagnostic["projected_xy"] = [list(point) for point in points]
    if not points:
        diagnostic["error"] = "finger_pad_projection_failed"
        return None, diagnostic
    arr = np.asarray(points, dtype=float)
    centroid = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    diagnostic["centroid_xy"] = list(centroid)
    return centroid, diagnostic


def robot_segmentation_ids(env: Any) -> List[int]:
    ids: List[int] = []
    robot_id = getattr(env, "segmentation_robot_id", None)
    if robot_id is not None:
        ids.append(int(robot_id) + 1)
    for instance_name, seg_id in dict(getattr(env, "instance_to_id", {}) or {}).items():
        lowered = str(instance_name).lower()
        if any(token in lowered for token in ("panda", "gripper", "robot")):
            ids.append(int(seg_id))
    return sorted(set(ids))


def gripper_robot_mask_centroid(seg: Any, env: Any) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
    import numpy as np

    ids = robot_segmentation_ids(env)
    diagnostic: Dict[str, Any] = {
        "method": "robot_mask_upper_frontier",
        "seg_ids": ids,
        "frontier_quantile": 0.20,
    }
    if not ids:
        diagnostic["error"] = "robot_segmentation_id_unavailable"
        return None, diagnostic
    mask = np.isin(np.asarray(seg), ids)
    ys, xs = np.where(mask)
    diagnostic["visible_pixels"] = int(xs.size)
    if xs.size == 0:
        diagnostic["error"] = "robot_mask_empty"
        return None, diagnostic

    y_cut = float(np.quantile(ys, 0.20))
    keep = ys <= y_cut
    if int(keep.sum()) < 8:
        y_cut = float(np.quantile(ys, 0.35))
        keep = ys <= y_cut
        diagnostic["frontier_quantile"] = 0.35
    if int(keep.sum()) == 0:
        diagnostic["error"] = "robot_frontier_empty"
        return None, diagnostic
    centroid = (float(xs[keep].mean()), float(ys[keep].mean()))
    diagnostic["frontier_pixels"] = int(keep.sum())
    diagnostic["centroid_xy"] = list(centroid)
    return centroid, diagnostic


def project_node_world_position(env: Any, camera_name: str, point_world: Any, width: int, height: int) -> Optional[Tuple[float, float]]:
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return None
    camera = camera_record(model, data, camera_name)
    if camera is None:
        return None
    return project_world_to_image(point_world, camera, width, height)


def extract_visibility_from_instance_segmentation(
    *,
    nodes: Iterable[Node],
    obs: Dict[str, Any],
    env: Any,
    camera_name: str,
    min_visible_pixels: int,
) -> Tuple[List[Node], Dict[str, Any]]:
    import numpy as np

    key = segmentation_key(obs, camera_name, "instance")
    diagnostics: Dict[str, Any] = {"segmentation_key": key, "method": "instance", "node_visibility": {}}
    if key is None:
        return list(nodes), diagnostics
    seg = np.asarray(obs[key])
    if seg.ndim == 3:
        seg = seg[..., 0]
    image_height, image_width = int(seg.shape[0]), int(seg.shape[1])
    instance_to_id = dict(getattr(env, "instance_to_id", {}) or {})
    robot_id = getattr(env, "segmentation_robot_id", None)
    out: List[Node] = []
    for node in nodes:
        world_point, world_diag = node_world_position(env, node.id, node.entity_type)
        world_xyz = None
        projected_xy = None
        if world_point is not None:
            world_xyz = tuple(float(value) for value in np.asarray(world_point, dtype=float).tolist())
            projected_xy = project_node_world_position(env, camera_name, world_point, image_width, image_height)
        if node.id == "gripper":
            centroid, gripper_projection_diag = gripper_robot_mask_centroid(seg, env)
            if centroid is None:
                centroid, gripper_projection_diag = gripper_finger_pad_centroid(
                    env=env,
                    camera_name=camera_name,
                    image_width=image_width,
                    image_height=image_height,
                )
            if centroid is None:
                centroid = projected_xy
                gripper_projection_diag = {"method": "grip_site_projection_fallback", "centroid_xy": list(centroid) if centroid else None}
            in_frame = (
                centroid is not None
                and 0.0 <= centroid[0] < float(image_width)
                and 0.0 <= centroid[1] < float(image_height)
            )
            pixels = int(min_visible_pixels) if in_frame else 0
            visible = bool(in_frame)
            diagnostics["node_visibility"][node.id] = {
                "seg_id": (int(robot_id) + 1 if robot_id is not None else None),
                "visible_pixels": pixels,
                "centroid_xy": list(centroid) if centroid else None,
                "visible": visible,
                "world_xyz": list(world_xyz) if world_xyz else None,
                "world_position": world_diag,
                "image_anchor": gripper_projection_diag,
            }
            out.append(
                Node(
                    id=node.id,
                    category=node.category,
                    entity_type=node.entity_type,
                    present=node.present,
                    visible=visible,
                    visible_pixels=pixels,
                    centroid_xy=centroid,
                )
            )
            continue
        else:
            seg_id = instance_to_id.get(node.id)
            mask = seg == seg_id if seg_id is not None else np.zeros_like(seg, dtype=bool)
        pixels, centroid = segmentation_stats(mask)
        visible = pixels >= min_visible_pixels
        graph_centroid = projected_xy if projected_xy is not None else centroid
        diagnostics["node_visibility"][node.id] = {
            "seg_id": (int(instance_to_id[node.id]) if node.id in instance_to_id else None),
            "visible_pixels": pixels,
            "segmentation_centroid_xy": list(centroid) if centroid else None,
            "centroid_xy": list(graph_centroid) if graph_centroid else None,
            "visible": visible,
            "world_xyz": list(world_xyz) if world_xyz else None,
            "world_position": world_diag,
        }
        out.append(
            Node(
                id=node.id,
                category=node.category,
                entity_type=node.entity_type,
                present=node.present,
                visible=visible,
                visible_pixels=pixels,
                centroid_xy=graph_centroid,
            )
        )
    return out, diagnostics


def model_name(model: Any, kind: str, idx: int) -> Optional[str]:
    method = getattr(model, f"{kind}_id2name", None)
    if method is None and kind == "camera":
        method = getattr(model, "camera_id2name", None)
    if callable(method):
        try:
            return method(idx)
        except Exception:
            return None
    return None


def model_names(model: Any, kind: str) -> List[str]:
    count_attr = {"body": "nbody", "geom": "ngeom", "site": "nsite", "camera": "ncam"}.get(kind, f"n{kind}")
    count = int(getattr(model, count_attr, 0) or 0)
    return [name for idx in range(count) if (name := model_name(model, kind, idx))]


def collect_mujoco_names(env: Any) -> Dict[str, List[str]]:
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    if model is None:
        return {"bodies": [], "geoms": [], "sites": [], "cameras": []}
    return {
        "bodies": model_names(model, "body"),
        "geoms": model_names(model, "geom"),
        "sites": model_names(model, "site"),
        "cameras": model_names(model, "camera"),
    }


def object_geom_map(env: Any, node_ids: Iterable[str]) -> Dict[str, List[str]]:
    names = collect_mujoco_names(env)["geoms"]
    mapping: Dict[str, List[str]] = {}
    for node_id in node_ids:
        norm = normalize_name(node_id)
        mapping[node_id] = sorted([geom for geom in names if normalize_name(geom).startswith(norm)])
    return mapping


def contact_records(env: Any) -> List[Dict[str, Any]]:
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return []
    records: List[Dict[str, Any]] = []
    for idx in range(int(getattr(data, "ncon", 0) or 0)):
        contact = data.contact[idx]
        records.append(
            {
                "contact_index": idx,
                "geom1": model_name(model, "geom", int(contact.geom1)),
                "geom2": model_name(model, "geom", int(contact.geom2)),
            }
        )
    return records


def save_rgb(obs: Dict[str, Any], path: Path, camera_name: str = "agentview") -> Optional[Path]:
    key = rgb_key(obs, camera_name)
    if key is None:
        return None
    try:
        import numpy as np
        from PIL import Image

        image = np.asarray(obs[key])
        if image.ndim == 3 and image.shape[-1] >= 3:
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image[..., :3].astype("uint8")).save(path)
            return path
    except Exception:
        return None
    return None
