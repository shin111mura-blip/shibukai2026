from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .canonicalize import make_config_hash, make_graph, sha256_payload
from .grasp_detector import FingerGeomSets, grasping_edges, infer_finger_geoms
from .node_extractor import (
    candidate_nodes_from_bddl,
    collect_mujoco_names,
    contact_records,
    ensure_repo_paths,
    extract_visibility_from_instance_segmentation,
    camera_record,
    object_geom_map,
    parse_bddl_problem,
    save_rgb,
)
from .relation_rules import observable_subset, structural_spatial_edges
from .schema import Edge, Node


def import_libero() -> Tuple[Any, Any, Any]:
    ensure_repo_paths()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    from libero.libero.envs.predicates.base_predicates import In, On

    return benchmark, get_libero_path, (SegmentationRenderEnv, On, In)


def resolve_task(suite: str, language_substring: str) -> Dict[str, Any]:
    benchmark, get_libero_path, _ = import_libero()
    task_suite = benchmark.get_benchmark_dict()[suite]()
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        language = getattr(task, "language", "")
        if language_substring.lower() in language.lower():
            return {
                "suite": suite,
                "task_id": task_id,
                "task_name": getattr(task, "name", str(task_id)),
                "instruction": language,
                "bddl_file": str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                "demo_file": str(Path(get_libero_path("datasets")) / task_suite.get_task_demonstration(task_id)),
            }
    raise ValueError(f"task not found in {suite}: {language_substring}")


def h5_attr_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def sorted_demo_keys(data_group: Any) -> List[str]:
    keys = [key for key in data_group.keys() if key.startswith("demo_")]
    return sorted(keys, key=lambda key: int(key.split("_")[-1]))


def patch_model_xml_paths(model_xml: str) -> str:
    ensure_repo_paths()
    from libero.libero import get_libero_path
    from libero.libero.utils import utils as libero_utils

    model_xml = libero_utils.postprocess_model_xml(model_xml, {})
    assets_root = Path(get_libero_path("assets"))
    root = ET.fromstring(model_xml)
    for elem in root.findall(".//*[@file]"):
        old_path = elem.get("file")
        if not old_path or "/assets/" not in old_path:
            continue
        local_path = assets_root / old_path.split("/assets/", 1)[1]
        if local_path.exists():
            elem.set("file", str(local_path))
    return ET.tostring(root, encoding="utf8").decode("utf8")


def create_env(bddl_file: str, image_size: int) -> Any:
    _benchmark, _get_libero_path, classes = import_libero()
    SegmentationRenderEnv, _On, _In = classes
    return SegmentationRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=image_size,
        camera_widths=image_size,
        camera_segmentations="instance",
    )


def refresh_segmentation_mapping(env: Any) -> None:
    """Populate SegmentationRenderEnv mappings without randomizing the scene."""
    inner = getattr(env, "env", env)
    model = getattr(inner, "model", None)
    instances = list(getattr(model, "instances_to_ids", {}).keys()) if model is not None else []
    if not hasattr(env, "segmentation_id_mapping"):
        return
    env.segmentation_id_mapping = {}
    env.segmentation_robot_id = None
    for idx, instance_name in enumerate(instances):
        if instance_name == "Panda0":
            env.segmentation_robot_id = idx
    for idx, instance_name in enumerate(instances):
        if instance_name not in ["Panda0", "RethinkMount0", "PandaGripper0"]:
            env.segmentation_id_mapping[idx] = instance_name
    env.instance_to_id = {v: k + 1 for k, v in env.segmentation_id_mapping.items()}


def set_demo_state(env: Any, state: Any, model_xml: Optional[str]) -> Dict[str, Any]:
    if model_xml:
        env.reset_from_xml_string(patch_model_xml_paths(model_xml))
        env.sim.reset()
        refresh_segmentation_mapping(env)
    obs = env.set_init_state(state)
    refresh_segmentation_mapping(env)
    return obs


def object_states(env: Any) -> Dict[str, Any]:
    inner = getattr(env, "env", env)
    return dict(getattr(inner, "object_states_dict", {}) or {})


def official_on_in_edges(env: Any, nodes: Iterable[Node]) -> Tuple[List[Edge], Dict[str, Any]]:
    _benchmark, _get_libero_path, classes = import_libero()
    _SegmentationRenderEnv, On, In = classes
    states = object_states(env)
    node_ids = [node.id for node in nodes if node.id != "gripper"]
    edges: List[Edge] = []
    diagnostics: Dict[str, Any] = {"on": {}, "inside": {}, "api": {"on": "On(subject, support) -> support.check_ontop(subject)", "in": "In(subject, container) -> container.check_contact(subject) and container.check_contain(subject)"}}
    on_pred = On()
    in_pred = In()
    for subject in node_ids:
        for obj in node_ids:
            if subject == obj or subject not in states or obj not in states:
                continue
            try:
                on_result = bool(on_pred(states[subject], states[obj]))
            except Exception as exc:
                on_result = False
                diagnostics["on"][f"{subject}->{obj}"] = f"{type(exc).__name__}: {exc}"
            if on_result:
                edges.append(Edge(subject, "on", obj))
            try:
                in_result = bool(in_pred(states[subject], states[obj]))
            except Exception as exc:
                in_result = False
                diagnostics["inside"][f"{subject}->{obj}"] = f"{type(exc).__name__}: {exc}"
            if in_result:
                edges.append(Edge(subject, "inside", obj))
                edges.append(Edge(obj, "contains", subject))
    return edges, diagnostics


def official_grasp_results(env: Any, object_ids: Iterable[str]) -> Dict[str, bool]:
    inner = getattr(env, "env", env)
    robot = None
    try:
        robot = inner.robots[0]
    except Exception:
        pass
    fn = getattr(inner, "_check_grasp", None)
    if not callable(fn):
        return {}
    results: Dict[str, bool] = {}
    for object_id in object_ids:
        try:
            obj = inner.get_object(object_id)
            results[object_id] = bool(fn(robot.gripper, obj))
        except Exception:
            continue
    return results


def world_positions_from_visibility(visibility_diag: Dict[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    positions: Dict[str, Tuple[float, float, float]] = {}
    for node_id, payload in visibility_diag.get("node_visibility", {}).items():
        value = payload.get("world_xyz")
        if value is None:
            continue
        positions[str(node_id)] = (float(value[0]), float(value[1]), float(value[2]))
    return positions


def image_plane_positions_from_visibility(
    visibility_diag: Dict[str, Any],
    *,
    image_size: int,
    rotate_180: bool,
) -> Dict[str, Tuple[float, float]]:
    positions: Dict[str, Tuple[float, float]] = {}
    for node_id, payload in visibility_diag.get("node_visibility", {}).items():
        value = payload.get("centroid_xy")
        if value is None:
            continue
        x = float(value[0])
        y = float(value[1])
        if rotate_180:
            x = float(image_size - 1) - x
            y = float(image_size - 1) - y
        positions[str(node_id)] = (x, y)
    return positions


def camera_depth_positions_from_visibility(
    env: Any,
    visibility_diag: Dict[str, Any],
    *,
    camera_name: str,
) -> Dict[str, float]:
    import numpy as np

    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return {}
    camera = camera_record(model, data, camera_name)
    if camera is None:
        return {}
    camera_pos = np.asarray(camera["pos_world"], dtype=float)
    camera_rot = np.asarray(camera["xmat"], dtype=float).reshape(3, 3)
    positions: Dict[str, float] = {}
    for node_id, payload in visibility_diag.get("node_visibility", {}).items():
        value = payload.get("world_xyz")
        if value is None:
            continue
        point = np.asarray(value, dtype=float)
        camera_frame = camera_rot.T @ (point - camera_pos)
        positions[str(node_id)] = float(-camera_frame[2])
    return positions


def build_frame_graphs(
    *,
    env: Any,
    obs: Dict[str, Any],
    bddl_file: Path,
    task_id: str,
    demo_id: str,
    frame_id: int,
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    min_visible_pixels = int(config.get("min_visible_pixels", 32))
    image_size = int(config.get("image_size", 256))
    nodes0 = candidate_nodes_from_bddl(bddl_file)
    nodes, visibility_diag = extract_visibility_from_instance_segmentation(
        nodes=nodes0,
        obs=obs,
        env=env,
        camera_name=str(config.get("camera_name", "agentview")),
        min_visible_pixels=min_visible_pixels,
    )
    contacts = contact_records(env)
    mujoco = collect_mujoco_names(env)
    geom_map = object_geom_map(env, [node.id for node in nodes if node.entity_type in {"object", "fixture"}])
    finger_geoms = infer_finger_geoms(mujoco["geoms"])
    official_grasps = official_grasp_results(env, geom_map)
    grasp_edges, grasp_diag = grasping_edges(contacts, geom_map, finger_geoms, official_grasps)
    official_edges, official_diag = official_on_in_edges(env, nodes)
    world_positions = world_positions_from_visibility(visibility_diag)
    image_plane_positions = image_plane_positions_from_visibility(
        visibility_diag,
        image_size=image_size,
        rotate_180=bool(config.get("relation_image_rotate_180", True)),
    )
    camera_depth_positions = camera_depth_positions_from_visibility(
        env,
        visibility_diag,
        camera_name=str(config.get("camera_name", "agentview")),
    )
    all_edges = (
        structural_spatial_edges(nodes, image_plane_positions, depth_positions=camera_depth_positions, visible_only=False)
        + official_edges
        + grasp_edges
    )
    config_hash = make_config_hash(config)
    world_graph = make_graph(
        source="rule_based",
        mode="world",
        task_id=task_id,
        demo_id=demo_id,
        frame_id=frame_id,
        nodes=nodes,
        edges=all_edges,
        config_hash=config_hash,
        extra_metadata={"camera_name": config.get("camera_name", "agentview")},
    )
    observable_nodes, observable_edges = observable_subset(nodes, all_edges)
    observable_graph = make_graph(
        source="rule_based",
        mode="observable",
        task_id=task_id,
        demo_id=demo_id,
        frame_id=frame_id,
        nodes=observable_nodes,
        edges=observable_edges,
        config_hash=config_hash,
        extra_metadata={"camera_name": config.get("camera_name", "agentview")},
    )
    diagnostics = {
        "frame_id": frame_id,
        "visibility": visibility_diag,
        "contact_diagnostics": grasp_diag,
        "contacts": contacts,
        "official_predicates": official_diag,
        "image_plane_positions": image_plane_positions,
        "camera_depth_positions": camera_depth_positions,
        "world_positions": world_positions,
        "finger_geoms": {"left": list(finger_geoms.left), "right": list(finger_geoms.right)},
        "mujoco_names": mujoco,
        "world_sha256": sha256_payload(world_graph),
        "observable_sha256": sha256_payload(observable_graph),
    }
    return world_graph, observable_graph, diagnostics


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def default_rule_config(image_size: int = 256) -> Dict[str, Any]:
    return {
        "rule_version": "0.7",
        "min_visible_pixels": 32,
        "image_size": image_size,
        "camera_name": "agentview",
        "teacher_graph": "rule_based_world_graph",
        "node_position_source": "mujoco_body_or_geom_world_xyz projected into upright agentview image plane; graph nodes omit coordinates",
        "relation_image_rotate_180": True,
        "spatial_edge_rule": "nearest_row_column_chain_on_upright_agentview_projection_plus_camera_depth_frontback",
        "forbidden_predicates": ["between", "touching", "near", "overlapping", "holding"],
    }


def run_demo(
    *,
    task: Dict[str, Any],
    demo_index: int,
    output_dir: Path,
    image_size: int = 256,
    max_frames: int | None = None,
) -> Dict[str, Any]:
    import h5py

    config = default_rule_config(image_size=image_size)
    bddl_file = Path(task["bddl_file"])
    demo_file = Path(task["demo_file"])
    env = create_env(str(bddl_file), image_size)
    world_hashes: List[str] = []
    obs_hashes: List[str] = []
    records = 0
    try:
        with h5py.File(demo_file, "r") as f:
            demo_keys = sorted_demo_keys(f["data"])
            demo_key = demo_keys[demo_index]
            group = f[f"data/{demo_key}"]
            states = group["states"][()]
            model_xml = h5_attr_text(group.attrs.get("model_file"))
            limit = len(states) if max_frames is None else min(len(states), max_frames)
            for frame_id in range(limit):
                obs = set_demo_state(env, states[frame_id], model_xml if frame_id == 0 else None)
                world_graph, observable_graph, diagnostics = build_frame_graphs(
                    env=env,
                    obs=obs,
                    bddl_file=bddl_file,
                    task_id=str(task["task_id"]),
                    demo_id=demo_key,
                    frame_id=frame_id,
                    config=config,
                )
                write_json(output_dir / "rule_based" / "world_graph" / demo_key / f"{frame_id:06d}.json", world_graph)
                write_json(output_dir / "rule_based" / "observable_graph" / demo_key / f"{frame_id:06d}.json", observable_graph)
                write_json(output_dir / "diagnostics" / demo_key / f"{frame_id:06d}.json", diagnostics)
                save_rgb(obs, output_dir / "frames" / demo_key / f"{frame_id:06d}.png")
                world_hashes.append(sha256_payload(world_graph))
                obs_hashes.append(sha256_payload(observable_graph))
                records += 1
    finally:
        try:
            env.close()
        except Exception:
            pass
    summary = {
        "task": task,
        "demo_index": demo_index,
        "demo_id": f"demo_{demo_index}",
        "records": records,
        "world_hashes": world_hashes,
        "observable_hashes": obs_hashes,
        "config": config,
    }
    write_json(output_dir / "reports" / "rule_generation_summary.json", summary)
    return summary
