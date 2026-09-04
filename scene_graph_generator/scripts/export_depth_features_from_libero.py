#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for rel in ("LIBERO", "openvla", "scripts/scene_graph"):
    path = str(ROOT / rel)
    if Path(path).exists() and path not in sys.path:
        sys.path.insert(0, path)


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def sha_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def h5_attr_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def sorted_demo_keys(data_group) -> list[str]:
    keys = [key for key in data_group.keys() if key.startswith("demo_")]
    return sorted(keys, key=lambda key: int(key.split("_")[-1]))


def local_hdf5_path(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    marker = "LIBERO/libero/datasets/"
    if marker in path:
        rel = path.split(marker, 1)[1]
        candidate = ROOT / "LIBERO" / "libero" / "datasets" / rel
        if candidate.exists():
            return candidate
    return p


def postprocess_model_xml(model_xml: str) -> str:
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
    return ET.tostring(root, encoding="utf8").decode("utf8")


def create_env(bddl_file: str, image_size: int, camera_name: str):
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=image_size,
        camera_widths=image_size,
        camera_names=[camera_name],
        camera_depths=True,
    )
    return env


def set_state(env, state, model_xml: str | None):
    if model_xml:
        env.reset_from_xml_string(postprocess_model_xml(model_xml))
        env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    getter = getattr(env, "_get_observations", None)
    if callable(getter):
        return getter()
    return env.set_init_state(state)


def normalize_name(value: str | None) -> str:
    import re

    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def model_names(model: Any, kind: str) -> list[str]:
    attr_names = {
        "body": "body_names",
        "geom": "geom_names",
        "site": "site_names",
        "camera": "camera_names",
    }
    names_attr = attr_names.get(kind)
    if names_attr and hasattr(model, names_attr):
        return [str(x) for x in getattr(model, names_attr) if x]
    attr = f"n{kind}"
    count = getattr(model, attr, 0)
    names = []
    for idx in range(int(count)):
        try:
            raw = model.id2name(idx, kind)
        except Exception:
            raw = None
        if raw:
            names.append(str(raw))
    return names


def name_to_id(model: Any, kind: str, name: str) -> int | None:
    method_names = [f"{kind}_name2id"]
    if kind == "camera":
        method_names.append("cam_name2id")
    for method_name in method_names:
        if not method_name:
            continue
        method = getattr(model, method_name, None)
        if callable(method):
            try:
                return int(method(name))
            except Exception:
                pass
    return None


def named_world_position(model: Any, data: Any, kind: str, name: str) -> np.ndarray | None:
    idx = name_to_id(model, kind, name)
    if idx is None:
        return None
    attr = {"body": "body_xpos", "geom": "geom_xpos", "site": "site_xpos"}.get(kind)
    if attr is None:
        return None
    try:
        return np.asarray(getattr(data, attr)[idx], dtype=np.float32)
    except Exception:
        return None


def prefixed_world_position(model: Any, data: Any, kind: str, node_id: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    target = normalize_name(node_id)
    names = model_names(model, kind)
    exact = [name for name in names if normalize_name(name) == target]
    prefixed = [name for name in names if normalize_name(name).startswith(target)]
    selected = exact or prefixed
    positions = []
    for name in selected:
        point = named_world_position(model, data, kind, name)
        if point is not None:
            positions.append(point)
    diagnostic = {"kind": kind, "matched_names": selected[:20], "num_matched_names": len(selected)}
    if not positions:
        return None, diagnostic
    return np.mean(np.stack(positions, axis=0), axis=0).astype(np.float32), diagnostic


def node_world_position(env: Any, node_id: str, entity_type: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return None, {"error": "missing_sim_model_or_data"}
    if node_id == "gripper" or entity_type == "gripper":
        point = named_world_position(model, data, "site", "gripper0_grip_site")
        return point, {"kind": "site", "matched_names": ["gripper0_grip_site"] if point is not None else []}
    for kind in ("body", "geom"):
        point, diagnostic = prefixed_world_position(model, data, kind, node_id)
        if point is not None:
            return point, diagnostic
    return None, {"error": "node_position_unavailable"}


def graph_with_xyz(graph: dict[str, Any], xyz_by_node: dict[str, list[float] | None]) -> dict[str, Any]:
    nodes = []
    for node in graph.get("nodes", []):
        enriched = dict(node)
        enriched["position_world_xyz"] = xyz_by_node.get(node["id"])
        nodes.append(enriched)
    return {**graph, "nodes": nodes, "graph_type": "3d_scene_graph", "coordinate_frame": "mujoco_world"}


def depth_to_feature(depth: np.ndarray, grid: int) -> tuple[np.ndarray, dict[str, Any]]:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    finite = np.isfinite(depth)
    clean = np.where(finite, depth, 0.0).astype(np.float32)
    h, w = clean.shape[:2]
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    pooled = []
    for yi in range(grid):
        for xi in range(grid):
            patch = clean[ys[yi] : ys[yi + 1], xs[xi] : xs[xi + 1]]
            valid = finite[ys[yi] : ys[yi + 1], xs[xi] : xs[xi + 1]]
            pooled.append(float(patch[valid].mean()) if valid.any() else 0.0)
    valid_values = clean[finite]
    stats = np.asarray(
        [
            float(valid_values.mean()) if valid_values.size else 0.0,
            float(valid_values.std()) if valid_values.size else 0.0,
            float(valid_values.min()) if valid_values.size else 0.0,
            float(valid_values.max()) if valid_values.size else 0.0,
            float(np.percentile(valid_values, 10)) if valid_values.size else 0.0,
            float(np.percentile(valid_values, 50)) if valid_values.size else 0.0,
            float(np.percentile(valid_values, 90)) if valid_values.size else 0.0,
            float(finite.mean()),
        ],
        dtype=np.float32,
    )
    vec = np.concatenate([np.asarray(pooled, dtype=np.float32), stats], axis=0)
    # Normalize per-frame while preserving the global stats at the tail.
    spatial = vec[:-8]
    scale = float(np.std(spatial)) if spatial.size else 0.0
    if scale > 1e-6:
        spatial = (spatial - float(np.mean(spatial))) / scale
    vec = np.concatenate([spatial.astype(np.float32), stats], axis=0)
    return vec.astype(np.float32), {
        "depth_shape": list(depth.shape),
        "depth_min": float(valid_values.min()) if valid_values.size else None,
        "depth_max": float(valid_values.max()) if valid_values.size else None,
        "depth_mean": float(valid_values.mean()) if valid_values.size else None,
        "depth_std": float(valid_values.std()) if valid_values.size else None,
        "finite_fraction": float(finite.mean()),
        "feature_dim": int(vec.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-manifest", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/feature_cache/all_frames/cache_manifest.jsonl"))
    ap.add_argument("--mapping", type=Path, default=Path("outputs/scene_graph_gt_openvla_spatial/reports/openvla_hdf5_mapping.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/depth_features/all_frames"))
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--camera-name", default="agentview")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--ontology", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/ontology/ontology.json"))
    ap.add_argument("--graph3d-output-dir", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/teacher_graph_3d/world_graph"))
    ap.add_argument("--limit-episodes", type=int, default=0)
    ap.add_argument("--limit-frames", type=int, default=0)
    args = ap.parse_args()

    report = {
        "status": "started",
        "input_manifest": str(args.input_manifest),
        "mapping": str(args.mapping),
        "image_size": args.image_size,
        "camera_name": args.camera_name,
        "grid": args.grid,
    }
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import h5py
        from safetensors.torch import save_file
        import torch
        from libero.libero import benchmark, get_libero_path
        from scene_graph_generator.graph_generator.schema import compact_graph, read_json

        mappings = {int(r["global_episode_index"]): r for r in read_jsonl(args.mapping)}
        ontology = json.loads(args.ontology.read_text())
        ordered_nodes = [node_id for node_id, _meta in sorted(ontology["nodes"].items(), key=lambda item: item[1]["index"])]
        rows = list(read_jsonl(args.input_manifest))
        if args.limit_frames:
            rows = rows[: args.limit_frames]
        by_episode = defaultdict(list)
        for row in rows:
            by_episode[int(row["global_episode_index"])].append(row)
        if args.limit_episodes:
            by_episode = dict(list(sorted(by_episode.items()))[: args.limit_episodes])

        task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        tensors: dict[str, torch.Tensor] = {}
        entries = []
        failures = []
        env = None
        last_env_key = None
        for episode_index, episode_rows in sorted(by_episode.items()):
            mapping = mappings.get(episode_index)
            if mapping is None:
                failures.append({"global_episode_index": episode_index, "error": "missing_hdf5_mapping"})
                continue
            hdf5_path = local_hdf5_path(mapping["source_hdf5"])
            if not hdf5_path.exists():
                failures.append({"global_episode_index": episode_index, "error": f"missing_hdf5:{hdf5_path}"})
                continue
            task = task_suite.get_task(int(mapping["libero_benchmark_task_id"]))
            bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env_key = (str(hdf5_path), bddl_file)
            if env_key != last_env_key:
                if env is not None:
                    env.close()
                env = create_env(bddl_file, args.image_size, args.camera_name)
                last_env_key = env_key
            with h5py.File(hdf5_path, "r") as h5:
                demo_group = h5[f"data/{mapping['libero_demo_key']}"]
                states = demo_group["states"]
                model_xml = h5_attr_text(demo_group.attrs.get("model_file"))
                frame_indices = mapping["frame_indices"]
                for row in sorted(episode_rows, key=lambda x: int(x["frame_index"])):
                    frame_index = int(row["frame_index"])
                    if frame_index >= len(frame_indices):
                        failures.append({**row, "error": "frame_index_out_of_alignment"})
                        continue
                    hdf5_frame = int(frame_indices[frame_index])
                    if hdf5_frame >= len(states):
                        failures.append({**row, "error": "hdf5_frame_out_of_range", "hdf5_frame": hdf5_frame})
                        continue
                    obs = set_state(env, states[hdf5_frame], model_xml if frame_index == 0 else None)
                    depth_key = f"{args.camera_name}_depth"
                    if depth_key not in obs:
                        depth_key = next((k for k in obs if "depth" in k and hasattr(obs[k], "shape")), None)
                    if depth_key is None:
                        failures.append({**row, "error": "depth_observation_missing", "obs_keys": sorted(obs)})
                        continue
                    vec, info = depth_to_feature(obs[depth_key], args.grid)
                    graph = compact_graph(read_json(Path(row["graph_path"])))
                    xyz_by_node: dict[str, list[float] | None] = {}
                    xyz_target = np.zeros((len(ordered_nodes), 3), dtype=np.float32)
                    xyz_mask = np.zeros((len(ordered_nodes),), dtype=np.float32)
                    xyz_diagnostics = {}
                    node_meta = {node["id"]: node for node in graph.get("nodes", [])}
                    for idx, node_id in enumerate(ordered_nodes):
                        meta = node_meta.get(node_id)
                        if meta is None:
                            xyz_by_node[node_id] = None
                            continue
                        point, diagnostic = node_world_position(env, node_id, meta["entity_type"])
                        xyz_diagnostics[node_id] = diagnostic
                        if point is None:
                            xyz_by_node[node_id] = None
                            continue
                        xyz = np.asarray(point, dtype=np.float32).reshape(3)
                        xyz_target[idx] = xyz
                        xyz_mask[idx] = 1.0
                        xyz_by_node[node_id] = xyz.astype(float).tolist()
                    graph3d = graph_with_xyz(graph, xyz_by_node)
                    graph3d_path = (
                        args.graph3d_output_dir
                        / f"task_{int(row['task_id']):02d}"
                        / f"global_{episode_index:06d}"
                        / f"{frame_index:06d}.json"
                    )
                    write_json(graph3d_path, graph3d)
                    key = row["sample_key"]
                    tensors[f"{key}__depth_features"] = torch.from_numpy(vec)
                    tensors[f"{key}__xyz_target"] = torch.from_numpy(xyz_target)
                    tensors[f"{key}__xyz_mask"] = torch.from_numpy(xyz_mask)
                    entries.append(
                        {
                            **row,
                            "depth_feature_key": f"{key}__depth_features",
                            "xyz_target_key": f"{key}__xyz_target",
                            "xyz_mask_key": f"{key}__xyz_mask",
                            "graph3d_path": str(graph3d_path),
                            "hdf5_frame_index": hdf5_frame,
                            "source_hdf5": str(hdf5_path),
                            "libero_demo_key": mapping["libero_demo_key"],
                            "depth_sha256": sha_array(np.asarray(obs[depth_key], dtype=np.float32)),
                            "depth_feature_sha256": sha_array(vec),
                            "xyz_mask_sum": int(xyz_mask.sum()),
                            "xyz_diagnostics": xyz_diagnostics,
                            **info,
                        }
                    )
        if env is not None:
            env.close()

        from safetensors.torch import load_file

        tmp_tensor = args.output_dir / "depth_features.safetensors.tmp"
        save_file(tensors, str(tmp_tensor), metadata={"format": "libero_agentview_depth_features", "grid": str(args.grid)})
        tensor_path = args.output_dir / "depth_features.safetensors"
        tmp_tensor.replace(tensor_path)
        reloaded = load_file(str(tensor_path), device="cpu")
        reload_ok = all(k in reloaded and sha_array(reloaded[k].numpy()) == sha_array(v.numpy()) for k, v in tensors.items())
        if not reload_ok:
            raise RuntimeError("depth feature reload/hash check failed")

        manifest_path = args.output_dir / "depth_manifest.jsonl"
        tmp_manifest = manifest_path.with_suffix(".jsonl.tmp")
        with open(tmp_manifest, "w") as f:
            for row in entries:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        tmp_manifest.replace(manifest_path)
        report.update(
            {
                "status": "ok" if not failures else "partial",
                "frames": len(entries),
                "requested_frames": sum(len(v) for v in by_episode.values()),
                "episodes": len(by_episode),
                "failures": failures[:20],
                "num_failures": len(failures),
                "depth_feature_dim": int(entries[0]["feature_dim"]) if entries else None,
                "graph3d_output_dir": str(args.graph3d_output_dir),
                "tensor_path": str(tensor_path),
                "manifest": str(manifest_path),
                "elapsed_sec": round(time.time() - started, 3),
            }
        )
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
    write_json(args.output_dir / "depth_feature_export_summary.json", report)
    print(json.dumps({"status": report["status"], "frames": report.get("frames"), "failures": report.get("num_failures"), "summary": str(args.output_dir / "depth_feature_export_summary.json")}, sort_keys=True))
    if report["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
