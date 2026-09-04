#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

from PIL import Image


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def tensor_sha256(tensor) -> str:
    import torch

    t = tensor.detach().cpu().contiguous()
    try:
        raw = t.view(torch.uint8).numpy().tobytes()
    except RuntimeError:
        raw = t.float().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats"))
    ap.add_argument("--export-manifest", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/rlds_frames/all_frames/manifest.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/feature_cache/all_frames"))
    ap.add_argument("--output-root", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial"))
    ap.add_argument("--feature-layer", type=int, default=-2)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.output_root / "reports"
    report = {"status": "started", "shard_size": args.shard_size, "feature_layer": args.feature_layer}
    try:
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import AutoModelForVision2Seq, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        rows = list(read_jsonl(args.export_manifest))
        if args.limit:
            rows = rows[: args.limit]
        torch.cuda.reset_peak_memory_stats()
        processor = AutoProcessor.from_pretrained(str(args.checkpoint), trust_remote_code=True, local_files_only=True)
        model = AutoModelForVision2Seq.from_pretrained(
            str(args.checkpoint),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        tokenizer = getattr(processor, "tokenizer", None)
        bos_id = getattr(tokenizer, "bos_token_id", None)

        def extract(row):
            image = Image.open(row["image_path"]).convert("RGB")
            prompt = f"In: What action should the robot take to {row['instruction'].lower()}?\nOut:"
            inputs = processor(prompt, image, return_tensors="pt")
            inputs = {
                k: v.to(device="cuda", dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to("cuda")
                for k, v in inputs.items()
            }
            with torch.inference_mode():
                out = model(
                    **inputs,
                    output_hidden_states=True,
                    output_projector_features=True,
                    return_dict=True,
                    use_cache=False,
                )
            projector = out.projector_features
            hidden = out.hidden_states[args.feature_layer]
            image_token_count = int(projector.shape[1])
            image_start = 1
            image_end = image_start + image_token_count
            input_ids = inputs["input_ids"][0].detach().cpu().tolist()
            attention = inputs["attention_mask"][0].detach().cpu().bool().tolist()
            bos_positions = [i for i, tok in enumerate(input_ids) if bos_id is not None and tok == bos_id] or [0]
            instruction_positions = [i + image_token_count for i, keep in enumerate(attention) if keep and i not in bos_positions]
            image_feat = hidden[:, image_start:image_end, :]
            instr_feat = hidden[:, instruction_positions, :]
            feat = torch.cat([image_feat, instr_feat], dim=1).squeeze(0).detach().cpu()
            token_type = torch.cat(
                [
                    torch.ones(image_feat.shape[1], dtype=torch.int64),
                    torch.full((instr_feat.shape[1],), 2, dtype=torch.int64),
                ]
            )
            attn = torch.ones(feat.shape[0], dtype=torch.bool)
            return feat, attn, token_type, {
                "image_token_count": int(image_feat.shape[1]),
                "instruction_token_count": int(instr_feat.shape[1]),
                "feature_shape": list(feat.shape),
            }

        started = time.time()
        completed_rows = []
        shard_count = math.ceil(len(rows) / args.shard_size)
        for shard_idx in range(shard_count):
            shard_rows = rows[shard_idx * args.shard_size : (shard_idx + 1) * args.shard_size]
            shard = args.output_dir / f"shard_{shard_idx:06d}.safetensors"
            shard_manifest = args.output_dir / f"shard_{shard_idx:06d}.jsonl"
            if shard.exists() and shard_manifest.exists():
                completed_rows.extend(list(read_jsonl(shard_manifest)))
                continue
            tensors = {}
            entries = []
            for local_idx, row in enumerate(shard_rows):
                global_idx = shard_idx * args.shard_size + local_idx
                feat, attn, token_type, info = extract(row)
                prefix = f"sample_{global_idx:06d}"
                tensors[f"{prefix}__features"] = feat
                tensors[f"{prefix}__attention_mask"] = attn
                tensors[f"{prefix}__token_type_mask"] = token_type
                entries.append(
                    {
                        **row,
                        "sample_key": prefix,
                        "shard": shard.name,
                        "feature_sha256": tensor_sha256(feat),
                        "attention_sha256": tensor_sha256(attn),
                        "token_type_sha256": tensor_sha256(token_type),
                        **info,
                    }
                )
            tmp_shard = shard.with_suffix(".safetensors.tmp")
            save_file(tensors, str(tmp_shard), metadata={"feature_layer": str(args.feature_layer), "format": "openvla_hidden_features_only"})
            tmp_shard.replace(shard)
            reloaded = load_file(str(shard), device="cpu")
            reload_ok = all(key in reloaded and tensor_sha256(reloaded[key]) == tensor_sha256(tensors[key]) for key in tensors)
            if not reload_ok:
                raise RuntimeError(f"Reload/hash check failed for {shard}")
            tmp_manifest = shard_manifest.with_suffix(".jsonl.tmp")
            with open(tmp_manifest, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry, sort_keys=True) + "\n")
            tmp_manifest.replace(shard_manifest)
            completed_rows.extend(entries)
            progress = {
                "status": "running",
                "completed_frames": len(completed_rows),
                "total_frames": len(rows),
                "completed_shards": shard_idx + 1,
                "total_shards": shard_count,
                "elapsed_sec": round(time.time() - started, 3),
            }
            write_json(args.output_dir / "progress.json", progress)
        manifest_path = args.output_dir / "cache_manifest.jsonl"
        tmp_manifest = manifest_path.with_suffix(".jsonl.tmp")
        with open(tmp_manifest, "w") as f:
            for row in completed_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        tmp_manifest.replace(manifest_path)
        keys = {(r["task_id"], r["global_episode_index"], r["frame_index"]) for r in completed_rows}
        report.update(
            {
                "status": "ok",
                "frames": len(completed_rows),
                "unique_frames": len(keys),
                "duplicates": len(completed_rows) - len(keys),
                "shards": shard_count,
                "elapsed_sec": round(time.time() - started, 3),
                "cache_dir": str(args.output_dir),
                "cache_manifest": str(manifest_path),
                "bytes": sum(p.stat().st_size for p in args.output_dir.glob("shard_*.safetensors")),
                "max_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "disk_free_bytes": shutil.disk_usage(args.output_dir).free,
                "feature_shapes": sorted({tuple(r["feature_shape"]) for r in completed_rows}),
            }
        )
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        blocker = [
            "# Blocker: Full Feature Cache",
            "",
            "## Cause",
            "Full feature cache generation failed.",
            "",
            "## Traceback",
            "```",
            report["traceback"],
            "```",
        ]
        (report_dir / "blocker_full_feature_cache.md").write_text("\n".join(blocker))
    write_json(report_dir / "full_feature_cache_summary.json", report)
    print(json.dumps({"status": report["status"], "frames": report.get("frames"), "report": str(report_dir / "full_feature_cache_summary.json")}, sort_keys=True))
    if report["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
