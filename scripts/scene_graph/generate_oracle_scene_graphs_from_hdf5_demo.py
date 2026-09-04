#!/usr/bin/env python3
"""Generate oracle scene graphs from LIBERO teleoperation HDF5 demos.

This requires the original LIBERO HDF5 files that contain flattened MuJoCo
states under data/demo_*/states. RLDS image/action caches are not enough for
oracle object-state graph generation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

from oracle_scene_graph_utils import (
    GraphThresholds,
    append_jsonl,
    ensure_repo_paths,
    import_libero_modules,
    make_graph_record,
    safe_json_dump,
    save_rgb_from_obs,
)


REQUESTED_BINARY_RELATIONS = {
    "left_of",
    "right_of",
    "front_of",
    "behind",
    "above",
    "below",
    "near",
    "touching",
    "grasped_by",
    "on",
    "inside",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--demo-index", type=int, default=0, help="Index of data/demo_* inside the HDF5 file.")
    parser.add_argument("--demo-file", type=Path, default=None, help="Optional explicit LIBERO *_demo.hdf5 path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hdf5_oracle_scene_graph/libero_spatial_task01_demo000"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--binary-only", action="store_true", help="Drop ternary or hyper-edge relations such as between.")
    return parser.parse_args()


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    keys = [key for key in data_group.keys() if key.startswith("demo_")]
    return sorted(keys, key=lambda key: int(key.split("_")[-1]))


def h5_attr_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def resolve_demo_file(suite: str, task_id: int, explicit: Optional[Path]) -> Tuple[Path, Dict[str, Any]]:
    ensure_repo_paths()
    benchmark, get_libero_path, _env_classes = import_libero_modules()
    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    metadata = {
        "suite": suite,
        "task_id": task_id,
        "task_name": getattr(task, "name", str(task_id)),
        "instruction": getattr(task, "language", ""),
        "problem_folder": getattr(task, "problem_folder", None),
        "bddl_file": os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
    }
    if explicit is not None:
        return explicit, metadata
    demo_rel = task_suite.get_task_demonstration(task_id)
    demo_file = Path(get_libero_path("datasets")) / demo_rel
    metadata["benchmark_demo_path"] = str(demo_file)
    return demo_file, metadata


def create_env_for_demo(demo_file: Path, metadata: Dict[str, Any], image_size: int):
    ensure_repo_paths()
    _benchmark, _get_libero_path, env_classes = import_libero_modules()
    OffScreenRenderEnv, _SegmentationRenderEnv = env_classes
    bddl_file = metadata["bddl_file"]
    with h5py.File(demo_file, "r") as f:
        h5_bddl = h5_attr_text(f["data"].attrs.get("bddl_file_name"))
        if h5_bddl and Path(h5_bddl).exists():
            bddl_file = h5_bddl
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=image_size,
        camera_widths=image_size,
    )
    return env


def set_demo_state(env: Any, state: Any, model_xml: Optional[str]) -> Any:
    if model_xml:
        import xml.etree.ElementTree as ET

        from libero.libero import get_libero_path
        from libero.libero.utils import utils as libero_utils

        model_xml = libero_utils.postprocess_model_xml(model_xml, {})
        assets_root = Path(get_libero_path("assets"))
        root = ET.fromstring(model_xml)
        for elem in root.findall(".//*[@file]"):
            old_path = elem.get("file")
            if not old_path or "/assets/" not in old_path:
                continue
            rel_path = old_path.split("/assets/", 1)[1]
            local_path = assets_root / rel_path
            if local_path.exists():
                elem.set("file", str(local_path))
        model_xml = ET.tostring(root, encoding="utf8").decode("utf8")
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        getter = getattr(env, "_get_observations", None)
        if callable(getter):
            return getter()
    return env.set_init_state(state)


def write_report(output_dir: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# HDF5 Oracle Scene Graph",
        "",
        f"- suite: `{summary['suite']}`",
        f"- task_id: `{summary['task_id']}`",
        f"- demo_key: `{summary['demo_key']}`",
        f"- source_hdf5: `{summary['source_hdf5']}`",
        f"- records: `{summary['records']}`",
        f"- graph_jsonl: `{summary['graph_jsonl']}`",
        f"- rgb_dir: `{summary['rgb_dir']}`",
        "",
        "Each record is generated by setting the MuJoCo simulator to the HDF5 demonstration state for the same timestep.",
    ]
    (output_dir / "README_hdf5_oracle_scene_graph.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def keep_binary_edges_only(record: Dict[str, Any]) -> Dict[str, Any]:
    thresholds = GraphThresholds()
    edges = requested_binary_edges(record, thresholds)
    removed = len(record.get("edges", [])) - len(
        [
            item
            for item in record.get("edges", [])
            if item.get("rel") in REQUESTED_BINARY_RELATIONS and not isinstance(item.get("dst"), list)
        ]
    )
    record["edges"] = dedupe_edges(edges)
    record.setdefault("metadata", {})["relation_scope"] = "binary_only"
    record["metadata"]["allowed_relations"] = sorted(REQUESTED_BINARY_RELATIONS)
    record["metadata"]["coordinate_convention"] = {
        "left_right_axis": "world_x",
        "front_behind_axis": "world_y",
        "above_below_axis": "world_z",
        "front_of_rule": "src.world_y < dst.world_y - margin",
    }
    record["metadata"]["removed_non_binary_edges"] = removed
    return record


def requested_binary_edges(record: Dict[str, Any], thresholds: GraphThresholds) -> list[Dict[str, Any]]:
    nodes = record.get("nodes", [])
    object_nodes = [n for n in nodes if n.get("type") == "object" and n.get("pos_world") is not None]
    edges: list[Dict[str, Any]] = []

    lateral_margin = 0.025
    depth_margin = 0.025
    vertical_margin = 0.025
    for src in object_nodes:
        for dst in object_nodes:
            if src.get("id") == dst.get("id"):
                continue
            sp = np.asarray(src["pos_world"], dtype=float)
            dp = np.asarray(dst["pos_world"], dtype=float)
            delta = sp - dp
            distance_xy = float(np.linalg.norm(delta[:2]))
            debug_base = {
                "delta_world": delta.tolist(),
                "distance_xy": distance_xy,
                "source": "mujoco_world_position",
            }
            if delta[0] < -lateral_margin:
                edges.append(make_edge(src["id"], "left_of", dst["id"], {**debug_base, "axis": "world_x", "margin": lateral_margin}))
            elif delta[0] > lateral_margin:
                edges.append(make_edge(src["id"], "right_of", dst["id"], {**debug_base, "axis": "world_x", "margin": lateral_margin}))
            if delta[1] < -depth_margin:
                edges.append(make_edge(src["id"], "front_of", dst["id"], {**debug_base, "axis": "world_y", "margin": depth_margin}))
            elif delta[1] > depth_margin:
                edges.append(make_edge(src["id"], "behind", dst["id"], {**debug_base, "axis": "world_y", "margin": depth_margin}))
            if delta[2] > vertical_margin:
                edges.append(make_edge(src["id"], "above", dst["id"], {**debug_base, "axis": "world_z", "margin": vertical_margin}))
            elif delta[2] < -vertical_margin:
                edges.append(make_edge(src["id"], "below", dst["id"], {**debug_base, "axis": "world_z", "margin": vertical_margin}))
            if distance_xy < thresholds.next_to:
                edges.append(make_edge(src["id"], "near", dst["id"], {**debug_base, "threshold": thresholds.next_to}))

    for item in record.get("edges", []):
        rel = item.get("rel")
        dst = item.get("dst")
        if isinstance(dst, list):
            continue
        if rel in {"touching", "on", "inside"}:
            copied = dict(item)
            copied["rel"] = rel
            edges.append(copied)
        elif rel == "grasping":
            edges.append(
                make_edge(
                    str(dst),
                    "grasped_by",
                    item.get("src"),
                    item.get("rule_debug", {}),
                    confidence=float(item.get("confidence", 1.0)),
                    source=item.get("source", "oracle_rule"),
                )
            )
    return edges


def make_edge(src: str, rel: str, dst: str, debug: Dict[str, Any], confidence: float = 1.0, source: str = "oracle_rule") -> Dict[str, Any]:
    return {"src": src, "rel": rel, "dst": dst, "confidence": confidence, "source": source, "rule_debug": debug}


def dedupe_edges(edges: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    deduped = []
    seen = set()
    for item in edges:
        key = json.dumps({"src": item.get("src"), "rel": item.get("rel"), "dst": item.get("dst")}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def main() -> None:
    args = parse_args()
    if args.sample_every <= 0:
        raise ValueError("--sample-every must be positive")

    demo_file, metadata = resolve_demo_file(args.suite, args.task_id, args.demo_file)
    if not demo_file.exists():
        raise FileNotFoundError(
            f"LIBERO HDF5 demo file not found: {demo_file}\n"
            "Download or place the original *_demo.hdf5 under LIBERO/libero/datasets, "
            "or pass --demo-file explicitly."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = args.output_dir / "graphs" / f"task_{args.task_id:02d}_demo_{args.demo_index:03d}.jsonl"
    if graph_path.exists():
        graph_path.unlink()

    env = None
    records = 0
    try:
        with h5py.File(demo_file, "r") as f:
            demo_keys = sorted_demo_keys(f["data"])
            if args.demo_index >= len(demo_keys):
                raise IndexError(f"--demo-index {args.demo_index} out of range; file has {len(demo_keys)} demos")
            demo_key = demo_keys[args.demo_index]
            demo_group = f[f"data/{demo_key}"]
            states = demo_group["states"][()]
            model_xml = h5_attr_text(demo_group.attrs.get("model_file"))
            instruction = metadata.get("instruction") or h5_attr_text(f["data"].attrs.get("language_instruction")) or ""

            env = create_env_for_demo(demo_file, metadata, args.image_size)
            limit = len(states) if args.max_steps is None else min(len(states), args.max_steps)
            for timestep in range(0, limit, args.sample_every):
                obs = set_demo_state(env, states[timestep], model_xml if timestep == 0 else None)
                if timestep > 0 and model_xml:
                    env.sim.set_state_from_flattened(states[timestep])
                    env.sim.forward()
                    getter = getattr(env, "_get_observations", None)
                    obs = getter() if callable(getter) else obs
                warnings = []
                record = make_graph_record(
                    suite=args.suite,
                    task_id=args.task_id,
                    task_name=metadata.get("task_name", str(args.task_id)),
                    instruction=instruction,
                    episode_id=args.demo_index,
                    timestep=timestep,
                    env=env,
                    thresholds=GraphThresholds(),
                    warnings=warnings,
                    camera_name=args.camera_name,
                    image_width=args.image_size,
                    image_height=args.image_size,
                )
                record["metadata"]["generator"] = Path(__file__).name
                record["metadata"]["source"] = "libero_teleop_hdf5_mujoco_state"
                record["metadata"]["source_hdf5"] = str(demo_file)
                record["metadata"]["demo_key"] = demo_key
                if args.binary_only:
                    record = keep_binary_edges_only(record)
                rgb_path = args.output_dir / "rgb" / f"task_{args.task_id:02d}" / f"demo_{args.demo_index:03d}" / f"t{timestep:06d}.png"
                saved_rgb = save_rgb_from_obs(obs, rgb_path)
                record["rgb_path"] = str(saved_rgb) if saved_rgb else None
                append_jsonl(graph_path, record)
                records += 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    summary = {
        "suite": args.suite,
        "task_id": args.task_id,
        "demo_index": args.demo_index,
        "demo_key": f"demo_{args.demo_index}",
        "source_hdf5": str(demo_file),
        "records": records,
        "graph_jsonl": str(graph_path),
        "rgb_dir": str(args.output_dir / "rgb"),
        "image_size": args.image_size,
        "sample_every": args.sample_every,
        "max_steps": args.max_steps,
        "binary_only": args.binary_only,
        "metadata": metadata,
    }
    safe_json_dump(summary, args.output_dir / "summary.json")
    write_report(args.output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
