#!/usr/bin/env python3
"""Export selected LIBERO RLDS frames and a manifest for YOLO preprocessing."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from PIL import Image


def _selected_indices(split_path: Path) -> set[int]:
    with open(split_path) as f:
        split = json.load(f)
    selected = set()
    for task in split["tasks"].values():
        selected.update(int(x) for x in task["selected_global_episode_indices"])
    return selected


def _task_lookup(split_path: Path) -> Dict[int, Dict[str, Any]]:
    with open(split_path) as f:
        split = json.load(f)
    lookup = {}
    for task in split["tasks"].values():
        for episode_idx in task["selected_global_episode_indices"]:
            lookup[int(episode_idx)] = task
    return lookup


def export_frames(data_root: Path, dataset_name: str, split_path: Path, output_root: Path, camera_name: str = "agentview") -> Path:
    selected = _selected_indices(split_path)
    task_lookup = _task_lookup(split_path)
    os.environ["OPENVLA_RLDS_DEMO_SPLIT_JSON"] = str(split_path)
    from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds

    ds, _ = make_dataset_from_rlds(
        name=dataset_name,
        data_dir=str(data_root),
        train=True,
        image_obs_keys={"primary": "image"},
        depth_obs_keys={},
        state_obs_keys=[],
        language_key="language_instruction",
    )
    manifest_path = output_root / "manifest.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as manifest:
        for episode in ds.as_numpy_iterator():
            observation = episode.get("observation", {})
            global_idx = int(observation["episode_index"][0])
            if global_idx not in selected:
                continue
            task = task_lookup[global_idx]
            task_id = int(task["task_id"])
            demo_id = int(global_idx)
            images = observation.get("image_primary")
            if images is None:
                raise KeyError("Could not find RGB image array in RLDS episode.")
            for image, timestep_value in zip(images, observation["timestep"]):
                timestep = int(timestep_value)
                if isinstance(image, (bytes, bytearray)):
                    image = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
                image_id = f"libero_spatial/task_{task_id:02d}/demo_{demo_id:03d}/step_{timestep:06d}/{camera_name}"
                rel_path = Path(image_id).with_suffix(".png")
                image_path = output_root / rel_path
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(image_path)
                height, width = image.shape[:2]
                row = {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "suite": "libero_spatial",
                    "task_id": task_id,
                    "task_name": task["task_name"],
                    "demo_id": demo_id,
                    "timestep": timestep,
                    "camera_name": camera_name,
                    "image_width": int(width),
                    "image_height": int(height),
                    "action_key": f"global_episode={demo_id}:timestep={timestep}",
                }
                manifest.write(json.dumps(row) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/modified_libero_rlds"))
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--split", type=Path, default=Path("splits/libero_spatial_10pct_seed42.json"))
    parser.add_argument("--output-root", type=Path, default=Path("cache/libero_frames"))
    args = parser.parse_args()
    manifest = export_frames(args.data_root, args.dataset_name, args.split, args.output_root)
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
