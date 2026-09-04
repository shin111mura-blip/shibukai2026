#!/usr/bin/env python3
"""Inspect LIBERO / robosuite / MuJoCo state availability for oracle graphs."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from oracle_scene_graph_utils import (
    camera_info,
    collect_candidate_object_names,
    create_libero_env,
    environment_versions,
    exception_payload,
    extract_object_nodes,
    get_contacts,
    get_model_data,
    list_model_names,
    make_graph_record,
    observation_summary,
    reset_env_to_episode,
    safe_json_dump,
    GraphThresholds,
)


def run_probe_command(command: List[str]) -> Dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def availability(label: str, value: Any) -> Dict[str, Any]:
    if value is None or value == [] or value == {}:
        return {"name": label, "available": False, "summary": None}
    summary: Any
    if isinstance(value, dict):
        summary = {"type": "dict", "count": len(value), "keys": sorted(str(k) for k in value.keys())[:50]}
    elif isinstance(value, (list, tuple)):
        summary = {"type": type(value).__name__, "count": len(value), "sample": value[:50]}
    else:
        summary = {"type": type(value).__name__, "repr": repr(value)[:300]}
    return {"name": label, "available": True, "summary": summary}


def write_readme(output_dir: Path, metadata: Dict[str, Any], inspection: Dict[str, Any]) -> None:
    lines = [
        "# LIBERO Oracle Scene Graph Environment Inspection",
        "",
        "## How to Re-run",
        "",
        "```bash",
        "PYTHONPATH=/workspace/openvla:/workspace/LIBERO python scripts/scene_graph/inspect_libero_state.py \\",
        "  --suite libero_spatial \\",
        "  --task-id 0 \\",
        "  --output-dir outputs/scene_graph_probe/env_inspection \\",
        "  --max-steps 5",
        "```",
        "",
        "For the local Docker setup in this repository:",
        "",
        "```bash",
        "docker compose run --rm libero-eval python scripts/scene_graph/inspect_libero_state.py --suite libero_spatial --task-id 0",
        "```",
        "",
        "## Environment",
        "",
        f"- Working directory: `{Path.cwd()}`",
        f"- Suite: `{metadata.get('suite')}`",
        f"- Task id: `{metadata.get('task_id')}`",
        f"- Task name: `{metadata.get('task_name')}`",
        f"- Instruction: `{metadata.get('instruction')}`",
        f"- BDDL file: `{metadata.get('bddl_file')}`",
        f"- Available benchmark names: `{', '.join(metadata.get('benchmark_names', []))}`",
        "",
        "## Availability Summary",
        "",
    ]
    for item in inspection.get("availability", []):
        marker = "yes" if item["available"] else "no"
        lines.append(f"- {item['name']}: {marker}")
    lines.extend(
        [
            "",
            "## Object Extraction",
            "",
            "Object nodes are extracted from LIBERO object registries when present, then backed off to MuJoCo body/geom name heuristics. Names matching robot, table, floor, wall, region, target, and workspace patterns are filtered out.",
            "",
            f"- Extracted node count: {len(inspection.get('sample_graph', {}).get('nodes', []))}",
            f"- Extracted relation count: {len(inspection.get('sample_graph', {}).get('edges', []))}",
            "",
            "## Current Limitations",
            "",
            "- `on` and `inside` use center-position and contact approximations unless MuJoCo extents are added later.",
            "- Camera intrinsics/extrinsics are recorded when model camera pose and fovy are exposed; projection may need calibration for renderer-specific conventions.",
            "- Segmentation/depth availability depends on constructing the env with `--camera-depths` and `--camera-segmentation`.",
            "",
        ]
    )
    if inspection.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in inspection["warnings"])
        lines.append("")
    (output_dir / "README_env_inspection.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_probe/env_inspection"))
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--camera-depths", action="store_true")
    parser.add_argument("--camera-segmentation", default=None, help="Example: instance, class, element")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shell_checks = {
        "pwd": run_probe_command(["pwd"]),
        "find_dirs": run_probe_command(["find", ".", "-maxdepth", "3", "-type", "d"]),
        "find_relevant_files": run_probe_command(["bash", "-lc", 'find . -maxdepth 4 -type f | grep -Ei "libero|robosuite|eval|benchmark|spatial|demo|hdf5|h5|env" | head -200']),
        "python_imports": run_probe_command(["python", "-c", "import libero, robosuite; print('libero ok'); print('robosuite', robosuite.__version__)"]),
    }

    warnings: List[str] = []
    metadata: Dict[str, Any] = {
        "working_directory": str(Path.cwd()),
        "libero_repo": str((Path.cwd() / "LIBERO").resolve()) if (Path.cwd() / "LIBERO").exists() else None,
        "openvla_repo": str((Path.cwd() / "openvla").resolve()) if (Path.cwd() / "openvla").exists() else None,
        "versions": environment_versions(),
        "shell_checks": shell_checks,
        "environment": {key: os.environ.get(key) for key in ["PYTHONPATH", "LIBERO_CONFIG_PATH", "MUJOCO_GL", "PYOPENGL_PLATFORM"]},
    }

    obs = None
    try:
        env, _task_suite, _task, init_states, task_metadata = create_libero_env(
            args.suite,
            args.task_id,
            image_size=args.image_size,
            camera_depths=args.camera_depths,
            camera_segmentations=args.camera_segmentation,
            seed=args.seed,
        )
        metadata.update(task_metadata)
        obs, reset_warnings = reset_env_to_episode(env, init_states, 0)
        warnings.extend(reset_warnings)
        for _ in range(args.max_steps):
            try:
                obs, _reward, _done, _info = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            except Exception as exc:
                warnings.append(f"zero-action step failed: {type(exc).__name__}: {exc}")
                break

        model, data = get_model_data(env)
        object_nodes, _object_geoms, geom_to_node = extract_object_nodes(env, warnings=warnings)
        contacts = get_contacts(env, geom_to_node)
        sample_graph = make_graph_record(
            suite=args.suite,
            task_id=args.task_id,
            task_name=task_metadata.get("task_name", str(args.task_id)),
            instruction=task_metadata.get("instruction", ""),
            episode_id=0,
            timestep=args.max_steps,
            env=env,
            thresholds=GraphThresholds(),
            warnings=warnings,
            image_width=args.image_size,
            image_height=args.image_size,
        )
        inspection = {
            "observation": observation_summary(obs),
            "env_attrs": sorted([name for name in dir(env) if not name.startswith("__")])[:300],
            "inner_env_attrs": sorted([name for name in dir(getattr(env, "env", env)) if not name.startswith("__")])[:300],
            "candidate_object_sources": collect_candidate_object_names(env, model),
            "camera_info": camera_info(env, obs),
            "contact_count": len(contacts),
            "sample_contacts": contacts[:50],
            "sample_graph": sample_graph,
            "availability": [
                availability("env", env),
                availability("env.env", getattr(env, "env", None)),
                availability("env.sim", getattr(env, "sim", None)),
                availability("env.sim.model", model),
                availability("env.sim.data", data),
                availability("observation", obs),
                availability("body_names", list_model_names(model, "body")),
                availability("geom_names", list_model_names(model, "geom")),
                availability("site_names", list_model_names(model, "site")),
                availability("joint_names", list_model_names(model, "joint")),
                availability("object_nodes", object_nodes),
                availability("contacts", contacts),
                availability("camera_info", camera_info(env, obs).get("cameras")),
            ],
            "warnings": warnings,
        }
        safe_json_dump(metadata, args.output_dir / "env_metadata.json")
        safe_json_dump(object_nodes, args.output_dir / "object_name_dump.json")
        safe_json_dump(list_model_names(model, "geom"), args.output_dir / "geom_name_dump.json")
        safe_json_dump(list_model_names(model, "body"), args.output_dir / "body_name_dump.json")
        safe_json_dump(camera_info(env, obs), args.output_dir / "camera_info.json")
        safe_json_dump({"contact_count": len(contacts), "contacts": contacts}, args.output_dir / "contact_probe.json")
        safe_json_dump(inspection, args.output_dir / "inspection_summary.json")
        write_readme(args.output_dir, metadata, inspection)
        env.close()
    except Exception as exc:
        metadata["fatal_error"] = exception_payload(exc)
        safe_json_dump(metadata, args.output_dir / "env_metadata.json")
        readme = args.output_dir / "README_env_inspection.md"
        readme.write_text(
            "# LIBERO Oracle Scene Graph Environment Inspection\n\n"
            "Environment creation or reset failed. See `env_metadata.json` for the full traceback.\n\n"
            f"Error: `{type(exc).__name__}: {exc}`\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
