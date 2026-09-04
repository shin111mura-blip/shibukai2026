#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
    t = tensor.detach().cpu().contiguous()
    try:
        raw = t.view(__import__("torch").uint8).numpy().tobytes()
    except RuntimeError:
        raw = t.float().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats"))
    ap.add_argument("--export-manifest", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/rlds_frames/smoke_100/manifest.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/feature_cache/smoke_100"))
    ap.add_argument("--output-root", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial"))
    ap.add_argument("--feature-layer", type=int, default=-2)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()
    report_dir = args.output_root / "reports"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "started", "feature_layer": args.feature_layer, "limit": args.limit}
    try:
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import AutoModelForVision2Seq, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        rows = list(read_jsonl(args.export_manifest))[: args.limit]
        if len(rows) != args.limit:
            raise RuntimeError(f"Expected {args.limit} rows, found {len(rows)}")
        torch.cuda.reset_peak_memory_stats()
        start_total = time.time()
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
                "prompt": prompt,
                "input_ids_shape": list(inputs["input_ids"].shape),
                "pixel_values_shape": list(inputs["pixel_values"].shape),
                "hidden_shape": list(hidden.shape),
                "image_token_count": int(image_feat.shape[1]),
                "instruction_token_count": int(instr_feat.shape[1]),
                "feature_shape": list(feat.shape),
            }

        tensors = {}
        entries = []
        per_frame_times = []
        deterministic = None
        for idx, row in enumerate(rows):
            t0 = time.time()
            feat, attn, token_type, info = extract(row)
            elapsed = time.time() - t0
            per_frame_times.append(elapsed)
            if idx == 0:
                feat2, _, _, _ = extract(row)
                deterministic = tensor_sha256(feat) == tensor_sha256(feat2)
            prefix = f"sample_{idx:06d}"
            tensors[f"{prefix}__features"] = feat
            tensors[f"{prefix}__attention_mask"] = attn
            tensors[f"{prefix}__token_type_mask"] = token_type
            entries.append(
                {
                    **row,
                    "sample_key": prefix,
                    "feature_sha256": tensor_sha256(feat),
                    "attention_sha256": tensor_sha256(attn),
                    "token_type_sha256": tensor_sha256(token_type),
                    **info,
                    "elapsed_sec": round(elapsed, 4),
                }
            )
        shard = args.output_dir / "shard_000000.safetensors"
        tmp = shard.with_suffix(".safetensors.tmp")
        save_file(tensors, str(tmp), metadata={"feature_layer": str(args.feature_layer), "format": "openvla_hidden_features_only"})
        tmp.replace(shard)
        reloaded = load_file(str(shard), device="cpu")
        reload_ok = all(key in reloaded and tensor_sha256(reloaded[key]) == tensor_sha256(tensors[key]) for key in tensors)
        manifest_path = args.output_dir / "cache_manifest.jsonl"
        tmp_manifest = manifest_path.with_suffix(".jsonl.tmp")
        with open(tmp_manifest, "w") as f:
            for row in entries:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        tmp_manifest.replace(manifest_path)
        total_sec = time.time() - start_total
        shard_bytes = shard.stat().st_size
        total_frames = 52970
        report.update(
            {
                "status": "ok" if reload_ok and deterministic else "failed",
                "num_entries": len(entries),
                "num_episodes": len({e["global_episode_index"] for e in entries}),
                "shard": str(shard),
                "manifest": str(manifest_path),
                "reload_ok": reload_ok,
                "deterministic_first_sample": deterministic,
                "nan_inf_ok": all(torch.isfinite(t).all().item() for k, t in tensors.items() if k.endswith("__features")),
                "total_sec": round(total_sec, 3),
                "avg_forward_sec": round(sum(per_frame_times) / len(per_frame_times), 4),
                "shard_bytes": shard_bytes,
                "avg_bytes_per_frame": int(shard_bytes / len(entries)),
                "estimated_full_cache_bytes": int(shard_bytes / len(entries) * total_frames),
                "estimated_full_cache_gib": round((shard_bytes / len(entries) * total_frames) / (1024**3), 3),
                "estimated_full_extraction_hours": round((sum(per_frame_times) / len(per_frame_times) * total_frames) / 3600, 3),
                "gpu_name": torch.cuda.get_device_name(0),
                "max_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "disk_free_bytes": shutil.disk_usage(args.output_dir).free,
                "feature_shapes": sorted({tuple(e["feature_shape"]) for e in entries}),
            }
        )
        if report["status"] != "ok":
            raise RuntimeError("Feature cache smoke checks failed")
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
    write_json(report_dir / "feature_cache_smoke.json", report)
    md = ["# Feature Cache Smoke 100", "", f"- Status: `{report['status']}`"]
    for key in [
        "num_entries",
        "num_episodes",
        "reload_ok",
        "deterministic_first_sample",
        "nan_inf_ok",
        "total_sec",
        "avg_forward_sec",
        "avg_bytes_per_frame",
        "estimated_full_cache_gib",
        "estimated_full_extraction_hours",
        "max_vram_bytes",
    ]:
        if key in report:
            md.append(f"- {key}: `{report[key]}`")
    if "traceback" in report:
        md.extend(["", "```", report["traceback"], "```"])
    (report_dir / "feature_cache_smoke.md").write_text("\n".join(md) + "\n")
    write_json(report_dir / "feature_cache_estimate.json", {k: report[k] for k in report if k.startswith("estimated_") or k in {"avg_bytes_per_frame", "disk_free_bytes", "num_entries", "status"}})
    print(json.dumps({"status": report["status"], "report": str(report_dir / "feature_cache_smoke.json")}, sort_keys=True))
    if report["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
