#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import sys
import traceback
from pathlib import Path


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "MISSING"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial"))
    ap.add_argument("--dataset-root", type=Path, default=Path("data/modified_libero_rlds"))
    ap.add_argument("--dataset-name", default="libero_spatial_no_noops")
    args = ap.parse_args()
    report_dir = args.output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    pkgs = [
        "tensorflow",
        "tensorflow-cpu",
        "tensorflow-datasets",
        "protobuf",
        "etils",
        "array-record",
        "absl-py",
        "apache-beam",
        "grpcio",
        "tensorstore",
        "dm-tree",
        "numpy",
        "tensorflow-metadata",
    ]
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "dataset_root": str(args.dataset_root),
        "dataset_name": args.dataset_name,
        "packages": {pkg: package_version(pkg) for pkg in pkgs},
        "steps": [],
        "status": "started",
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def step(name: str, fn):
        try:
            value = fn()
            report["steps"].append({"name": name, "status": "ok", "value": value})
            return value
        except Exception:
            report["steps"].append({"name": name, "status": "error", "traceback": traceback.format_exc()})
            report["status"] = "failed"
            raise

    try:
        google_protobuf = step("import google.protobuf", lambda: __import__("google.protobuf").protobuf.__version__)
        tf = step("import tensorflow", lambda: getattr(__import__("tensorflow"), "__version__", None))
        try:
            import tensorflow as tf_mod

            tf_mod.config.set_visible_devices([], "GPU")
            report["tensorflow_gpu_policy"] = "set_visible_devices([], GPU) succeeded"
        except RuntimeError as exc:
            report["tensorflow_gpu_policy"] = f"not changed after initialization: {exc}"
        report["tensorflow_version"] = tf
        tfds = step("import tensorflow_datasets", lambda: getattr(__import__("tensorflow_datasets"), "__version__", None))
        report["tfds_version"] = tfds

        def builder_probe():
            import tensorflow_datasets as tfds_mod

            return str(tfds_mod.builder(args.dataset_name, data_dir=str(args.dataset_root)))

        report["builder"] = step("tfds builder", builder_probe)

        def one_episode_probe():
            import tensorflow_datasets as tfds_mod

            ds = tfds_mod.builder(args.dataset_name, data_dir=str(args.dataset_root)).as_dataset(split="train")
            ep = next(iter(tfds_mod.as_numpy(ds.take(1))))
            steps = ep["steps"]
            first = next(iter(steps))
            obs = first["observation"]
            image = obs.get("image")
            if image is None:
                image = obs.get("image_primary")
            instruction = first.get("language_instruction", None)
            if instruction is None and "task" in first:
                instruction = first["task"].get("language_instruction", None)
            if instruction is None and "language_instruction" in ep:
                instruction = ep["language_instruction"]
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8")
            episode_index = obs.get("episode_index", ep.get("episode_index", -1))
            timestep = obs.get("timestep", first.get("timestep", 0))
            return {
                "global_episode_index": int(episode_index),
                "frame_index": int(timestep),
                "image_shape": list(image.shape),
                "instruction": str(instruction),
                "top_level_keys": sorted(str(k) for k in ep.keys()),
                "first_step_keys": sorted(str(k) for k in first.keys()),
                "observation_keys": sorted(str(k) for k in obs.keys()),
            }

        report["first_episode"] = step("read one RLDS episode", one_episode_probe)
        report["status"] = "ok"
    except Exception:
        pass

    json_path = report_dir / "rlds_runtime_validation.json"
    md_path = report_dir / "rlds_runtime_validation.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# RLDS Runtime Validation", "", f"- Status: `{report['status']}`", ""]
    lines.append("## Packages")
    for pkg, version in report["packages"].items():
        lines.append(f"- `{pkg}`: `{version}`")
    lines.append("")
    for item in report["steps"]:
        lines.append(f"## {item['name']}")
        lines.append(f"- Status: `{item['status']}`")
        if "value" in item:
            lines.append("```")
            lines.append(str(item["value"]))
            lines.append("```")
        if "traceback" in item:
            lines.append("```")
            lines.append(item["traceback"])
            lines.append("```")
        lines.append("")
    md_path.write_text("\n".join(lines))
    print(json.dumps({"status": report["status"], "json": str(json_path), "md": str(md_path)}, sort_keys=True))
    if report["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
