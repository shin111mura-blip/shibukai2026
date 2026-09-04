#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from scene_graph.node_extractor import parse_bddl_problem
from scene_graph.rule_generator import resolve_task, sorted_demo_keys


TARGET = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-language", default=TARGET)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_gt_v0"))
    return parser.parse_args()


def module_status(name: str) -> dict:
    try:
        module = __import__(name)
        return {"available": True, "version": getattr(module, "__version__", None), "file": getattr(module, "__file__", None)}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = args.output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    task = resolve_task(args.suite, args.task_language)
    bddl = parse_bddl_problem(Path(task["bddl_file"]))
    hdf5 = {"available": False}
    try:
        import h5py

        with h5py.File(task["demo_file"], "r") as f:
            demos = sorted_demo_keys(f["data"])
            first = f[f"data/{demos[0]}"]
            hdf5 = {
                "available": True,
                "demo_count": len(demos),
                "first_demo": demos[0],
                "first_demo_attrs": {k: str(v) for k, v in first.attrs.items()},
                "datasets": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in first.items() if hasattr(v, "shape")},
                "data_attrs": {k: str(v) for k, v in f["data"].attrs.items()},
            }
    except Exception as exc:
        hdf5 = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    report = {
        "target_task": task,
        "bddl": {
            "objects": bddl["objects"],
            "fixtures": bddl["fixtures"],
            "objects_of_interest": bddl["obj_of_interest"],
            "language": bddl["language"],
        },
        "hdf5": hdf5,
        "environment": {
            "python": platform.python_version(),
            "modules": {name: module_status(name) for name in ["libero", "robosuite", "mujoco", "h5py", "torch", "transformers"]},
            "output_path_note": "/sandbox is absent in this workspace; using outputs/scene_graph_gt_v0.",
        },
        "confirmed_apis_from_source": {
            "On": "LIBERO base_predicates.On(subject, support) returns support.check_ontop(subject).",
            "In": "LIBERO base_predicates.In(subject, container) returns container.check_contact(subject) and container.check_contain(subject).",
            "SegmentationRenderEnv": "LIBERO env_wrapper.SegmentationRenderEnv exposes instance_to_id and segmentation observation keys.",
            "grasping": "Uses robosuite/LIBERO _check_grasp when present for diagnostics, and v0 binary contact rule for graph edge generation.",
        },
    }
    json_path = reports / "investigation.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Scene Graph GT v0 Investigation",
        "",
        f"- Suite: `{task['suite']}`",
        f"- Task ID: `{task['task_id']}`",
        f"- Task name: `{task['task_name']}`",
        f"- Instruction: `{task['instruction']}`",
        f"- BDDL: `{task['bddl_file']}`",
        f"- Demonstration HDF5: `{task['demo_file']}`",
        f"- Demonstration count: `{hdf5.get('demo_count', 'unavailable')}`",
        f"- HDF5 available: `{hdf5.get('available')}`",
        "",
        "## Objects",
        "",
        *[f"- `{name}`: `{category}`" for name, category in sorted(bddl["objects"].items())],
        "",
        "## Fixtures",
        "",
        *[f"- `{name}`: `{category}`" for name, category in sorted(bddl["fixtures"].items())],
        "",
        "## Confirmed APIs",
        "",
        "- `On(subject, support)` is implemented as `support.check_ontop(subject)`.",
        "- `In(subject, container)` is implemented as `container.check_contact(subject) and container.check_contain(subject)`.",
        "- `SegmentationRenderEnv` provides instance segmentation mapping after reset.",
        "- `between` and `touching` are excluded from canonical graph output.",
        "",
        "## Environment Gap",
        "",
        "- `/sandbox` is absent here, so outputs are written under `outputs/scene_graph_gt_v0`.",
    ]
    (reports / "investigation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"investigation": str(reports / "investigation.md"), "task": task}, indent=2))


if __name__ == "__main__":
    main()
