#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph_generator.graph_generator.feature_extractor import OpenVLAFeatureExtractor
from scene_graph_generator.graph_generator.schema import file_sha256, write_json


def is_hf_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def resolve_checkpoint(path: Path) -> Path:
    if is_hf_checkpoint(path):
        return path
    candidates = []
    import os

    for env_name in ("OPENVLA_BASE_CHECKPOINT", "OPENVLA_BASE_PATH"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    candidates.extend(
        [
            path,
            Path("/sandbox/graph_internalization_exp/openvla_graph_internalization_rgb_runtime_complete_v3/checkpoints/openvla_7b_base"),
            Path("/sandbox/checkpoints/openvla_7b_base"),
            Path("/sandbox/checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats"),
            Path("/sandbox/openvla/checkpoints/openvla_7b_base"),
            Path("/sandbox/openvla/checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats"),
        ]
    )
    for candidate in candidates:
        expanded = candidate.expanduser()
        if is_hf_checkpoint(expanded):
            return expanded
    checked = "\n".join(f"- {candidate.expanduser()}" for candidate in candidates)
    raise RuntimeError(
        "Could not locate a complete local Base OpenVLA checkpoint with config.json. "
        "Set OPENVLA_BASE_CHECKPOINT to the directory containing config.json and model shards. "
        f"Checked:\n{checked}"
    )


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            f.write("\n")


def resolve_episode_dir(value: str, data_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        if path.exists():
            return path
        parts = path.parts
        data_root_name = data_root.name
        if data_root_name in parts:
            suffix = Path(*parts[parts.index(data_root_name) + 1 :])
            candidate = data_root / suffix
            if candidate.exists():
                return candidate
        if "episodes" in parts:
            suffix = Path(*parts[parts.index("episodes") + 1 :])
            candidate = data_root / "episodes" / suffix
            if candidate.exists():
                return candidate
        return path
    if path.exists():
        return path
    candidate = data_root.parent.parent / path
    if candidate.exists():
        return candidate
    return data_root / path


def raw_instruction_from_metadata(meta: dict[str, Any]) -> str:
    raw = meta.get("raw_instruction") or meta.get("language_instruction")
    if raw:
        return str(raw)
    formatted = str(meta.get("formatted_prompt") or "")
    prefix = "In: What action should the robot take to "
    suffix = "\nOut:"
    if formatted.startswith(prefix) and formatted.endswith(suffix):
        return formatted[len(prefix) : -len(suffix)]
    task_name = str(meta.get("task_name") or "").replace("_", " ").strip()
    if task_name:
        return task_name
    raise RuntimeError(f"Cannot recover instruction for episode {meta.get('episode_id')}")


def sample_key(meta: dict[str, Any], frame_index: int) -> str:
    return (
        f"{meta.get('suite_name', 'libero_spatial')}/"
        f"task_{int(meta['task_id']):02d}/"
        f"init_{int(meta.get('initial_state_id', -1)):04d}/"
        f"{meta['policy_id']}/"
        f"{meta['episode_id']}/"
        f"step_{frame_index:06d}"
    )


def save_shard(path: Path, payload: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(payload, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Frozen Base OpenVLA token features for rollout frames.")
    parser.add_argument("--data-root", type=Path, default=Path("data/openvla_rollout_graph_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/openvla_rollout_graph_v2/openvla_feature_cache/base_openvla_rollout_v1"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-layer", type=int, default=-2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--limit-episodes-per-split", type=int, default=None)
    parser.add_argument("--limit-frames-per-split", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--log-every-frames", type=int, default=1000)
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    args.checkpoint = resolve_checkpoint(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_shards == 1:
        manifest_path = args.output_dir / "cache_manifest.jsonl"
        shard_prefix = ""
        summary_name = "cache_summary.json"
    else:
        manifest_path = args.output_dir / f"cache_manifest.part_{args.shard_index:02d}.jsonl"
        shard_prefix = f"part_{args.shard_index:02d}_"
        summary_name = f"cache_summary.part_{args.shard_index:02d}.json"
    if manifest_path.exists():
        raise RuntimeError(f"{manifest_path} already exists. Move it aside before rebuilding the cache.")

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(f"Pillow is required in the server environment: {exc}") from exc

    extractor = OpenVLAFeatureExtractor(args.checkpoint, device=args.device, dtype=args.dtype)
    env_report = extractor.environment_report()
    shard_payload: dict[str, Any] = {}
    shard_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    shard_index = 0
    frames_written = 0
    episodes_seen = 0
    started = time.time()

    def log(message: str) -> None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

    def flush() -> None:
        nonlocal shard_payload, shard_rows, shard_index
        if not shard_rows:
            return
        shard_name = f"{shard_prefix}features_{shard_index:06d}.safetensors"
        save_shard(args.output_dir / shard_name, shard_payload)
        for row in shard_rows:
            row["shard"] = shard_name
        append_jsonl(manifest_path, shard_rows)
        shard_payload = {}
        shard_rows = []
        shard_index += 1

    def process_pending() -> None:
        nonlocal frames_written
        if not pending:
            return
        images = [row.pop("_image") for row in pending]
        instructions = [row.pop("_instruction") for row in pending]
        features, attn, token_type = extractor.extract_batch(images, instructions, feature_layer=args.feature_layer)
        for i, row in enumerate(pending):
            key = row["sample_key"]
            n = int(attn[i].sum().detach().cpu())
            shard_payload[f"{key}__features"] = features[i, :n].detach().cpu().to(extractor.torch.bfloat16)
            shard_payload[f"{key}__attention_mask"] = attn[i, :n].detach().cpu().bool()
            shard_payload[f"{key}__token_type_mask"] = token_type[i, :n].detach().cpu().long()
            row["feature_shape"] = list(shard_payload[f"{key}__features"].shape)
            shard_rows.append(row)
            frames_written += 1
            if len(shard_rows) >= args.shard_size:
                flush()
        if args.log_every_frames > 0 and (frames_written % args.log_every_frames < len(pending) or frames_written == len(pending)):
            elapsed = max(time.time() - started, 1e-6)
            log(f"cached frames={frames_written} shards={shard_index} rate={frames_written / elapsed:.2f} frame/s")
        pending.clear()

    for split in args.splits:
        log(f"split {split} started")
        split_episodes = 0
        split_frames = 0
        split_manifest = args.data_root / "manifests" / f"{split}.jsonl"
        for manifest_episode_index, ep_row in enumerate(read_jsonl(split_manifest)):
            if manifest_episode_index % args.num_shards != args.shard_index:
                continue
            if args.limit_episodes is not None and episodes_seen >= args.limit_episodes:
                break
            if args.limit_episodes_per_split is not None and split_episodes >= args.limit_episodes_per_split:
                break
            episode_dir = resolve_episode_dir(str(ep_row["episode_dir"]), args.data_root)
            meta = read_json(episode_dir / "metadata.json")
            frames_path = episode_dir / "frames.npz"
            arrays = np.load(frames_path)
            rgb = arrays["rgb"]
            instruction = raw_instruction_from_metadata(meta)
            episodes_seen += 1
            split_episodes += 1
            frame_indices = range(0, int(rgb.shape[0]), args.frame_stride)
            for frame_index in frame_indices:
                scheduled_frames = frames_written + len(pending)
                if args.limit_frames is not None and scheduled_frames >= args.limit_frames:
                    break
                if args.limit_frames_per_split is not None and split_frames >= args.limit_frames_per_split:
                    break
                image = Image.fromarray(rgb[frame_index])
                key = sample_key(meta, frame_index)
                pending.append(
                    {
                        "sample_key": key,
                        "split": split,
                        "episode_dir": str(episode_dir),
                        "frames_npz": str(frames_path),
                        "frame_index": int(frame_index),
                        "task_id": int(meta["task_id"]),
                        "initial_state_id": int(meta.get("initial_state_id", -1)),
                        "policy_id": str(meta["policy_id"]),
                        "episode_id": str(meta["episode_id"]),
                        "episode_success": bool(meta.get("episode_success", False)),
                        "failure_category": str(meta.get("failure_category", "")),
                        "terminal_reason": str(meta.get("terminal_reason", "")),
                        "feature_layer": args.feature_layer,
                        "checkpoint": str(args.checkpoint),
                        "_image": image,
                        "_instruction": instruction,
                    }
                )
                split_frames += 1
                if len(pending) >= args.batch_size:
                    process_pending()
            if args.limit_frames is not None and frames_written + len(pending) >= args.limit_frames:
                break
            if args.limit_frames_per_split is not None and split_frames >= args.limit_frames_per_split:
                break
        if args.limit_episodes is not None and episodes_seen >= args.limit_episodes:
            break
        if args.limit_frames is not None and frames_written >= args.limit_frames:
            break

    process_pending()
    flush()
    summary = {
        "status": "ok",
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "cache_manifest": str(manifest_path),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint / "config.json") if (args.checkpoint / "config.json").exists() else None,
        "feature_layer": args.feature_layer,
        "shard_size": args.shard_size,
        "num_shards": shard_index,
        "episodes_seen": episodes_seen,
        "frames_written": frames_written,
        "frame_stride": args.frame_stride,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "manifest_part": str(manifest_path),
        "limit_episodes_per_split": args.limit_episodes_per_split,
        "limit_frames_per_split": args.limit_frames_per_split,
        "extractor_environment": env_report,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(args.output_dir / summary_name, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
