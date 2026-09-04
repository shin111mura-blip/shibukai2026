#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha_array(arr) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/manifests/train_frames.jsonl"))
    ap.add_argument("--dataset-root", type=Path, default=Path("data/modified_libero_rlds"))
    ap.add_argument("--dataset-name", default="libero_spatial_no_noops")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/rlds_frames/smoke_100"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-frames-per-episode", type=int, default=25)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow_datasets as tfds

    wanted = defaultdict(set)
    wanted_rows = {}
    per_episode_counts = defaultdict(int)
    for row in read_jsonl(args.manifest):
        if len(wanted_rows) >= args.limit:
            break
        ep = int(row["global_episode_index"])
        fr = int(row["frame_index"])
        if per_episode_counts[ep] >= args.max_frames_per_episode:
            continue
        wanted[ep].add(fr)
        wanted_rows[(ep, fr)] = row
        per_episode_counts[ep] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds = tfds.builder(args.dataset_name, data_dir=str(args.dataset_root)).as_dataset(split="train")
    exported = []
    for global_idx, ep in enumerate(tfds.as_numpy(ds)):
        if global_idx not in wanted:
            if global_idx > max(wanted):
                break
            continue
        steps = ep["steps"]
        for frame_idx, step in enumerate(steps):
            if frame_idx not in wanted[global_idx]:
                continue
            image = step["observation"]["image"]
            instruction = step["language_instruction"]
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8")
            rel = Path(f"task_{wanted_rows[(global_idx, frame_idx)]['task_id']:02d}") / f"global_{global_idx:06d}" / f"{frame_idx:06d}.png"
            path = args.output_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".png.tmp")
            Image.fromarray(image).save(tmp, format="PNG")
            tmp.replace(path)
            png_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            exported.append(
                {
                    **wanted_rows[(global_idx, frame_idx)],
                    "image_path": str(path),
                    "image_array_sha256": sha_array(image),
                    "png_sha256": png_hash,
                    "image_shape": list(image.shape),
                    "instruction_from_rlds": instruction,
                    "instruction_matches_manifest": instruction == wanted_rows[(global_idx, frame_idx)]["instruction"],
                }
            )
        if len(exported) >= len(wanted_rows):
            break
    missing = sorted(set(wanted_rows) - {(int(r["global_episode_index"]), int(r["frame_index"])) for r in exported})
    atomic_write_json(
        args.output_dir / "export_summary.json",
        {
            "status": "ok" if not missing else "failed",
            "requested": len(wanted_rows),
            "exported": len(exported),
            "missing": missing[:20],
            "format": "PNG lossless",
            "dataset_name": args.dataset_name,
        },
    )
    manifest_path = args.output_dir / "manifest.jsonl"
    tmp_manifest = manifest_path.with_suffix(".jsonl.tmp")
    with open(tmp_manifest, "w") as f:
        for row in exported:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    tmp_manifest.replace(manifest_path)
    print(json.dumps({"exported": len(exported), "missing": len(missing), "manifest": str(manifest_path)}, sort_keys=True))
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
