#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
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


def parameter_sha256(tensor) -> str:
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
    ap.add_argument("--export-manifest", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/rlds_frames/smoke_100/manifest.jsonl"))
    ap.add_argument("--output-root", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial"))
    ap.add_argument("--feature-layer", type=int, default=-2)
    args = ap.parse_args()
    report_dir = args.output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "started",
        "checkpoint": str(args.checkpoint),
        "feature_layer": args.feature_layer,
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch
        import transformers
        from transformers import AutoModelForVision2Seq, AutoProcessor

        row = next(read_jsonl(args.export_manifest))
        image = Image.open(row["image_path"]).convert("RGB")
        instruction = row["instruction"]
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("torch.cuda.is_available() is False; refusing to load OpenVLA 7B for this phase.")
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        processor = AutoProcessor.from_pretrained(str(args.checkpoint), trust_remote_code=True, local_files_only=True)
        model = AutoModelForVision2Seq.from_pretrained(
            str(args.checkpoint),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        first_name, first_param = next(iter(model.named_parameters()))
        before_hash = parameter_sha256(first_param)
        inputs = processor(prompt, image, return_tensors="pt")
        inputs = {
            k: v.to(device=device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(device)
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
        attention = inputs["attention_mask"][0].detach().cpu().bool().tolist()
        tokenizer = getattr(processor, "tokenizer", None)
        bos_id = getattr(tokenizer, "bos_token_id", None)
        input_ids = inputs["input_ids"][0].detach().cpu().tolist()
        bos_positions = [i for i, tok in enumerate(input_ids) if bos_id is not None and tok == bos_id] or [0]
        instruction_positions = [
            i + image_token_count
            for i, keep in enumerate(attention)
            if keep and i not in bos_positions
        ]
        image_features = hidden[:, image_start:image_end, :]
        instruction_features = hidden[:, instruction_positions, :]
        after_hash = parameter_sha256(first_param)
        grads_none = all(p.grad is None for p in model.parameters())
        report.update(
            {
                "status": "ok",
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "torch_dtype": "bfloat16",
                "device": device,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_capability": list(torch.cuda.get_device_capability(0)),
                "max_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "model_class": type(model).__name__,
                "processor_class": type(processor).__name__,
                "tokenizer_class": type(tokenizer).__name__ if tokenizer is not None else None,
                "model_config_model_type": getattr(model.config, "model_type", None),
                "load_and_forward_sec": round(time.time() - started, 3),
                "sample": {
                    "task_id": row["task_id"],
                    "global_episode_index": row["global_episode_index"],
                    "frame_index": row["frame_index"],
                    "image_path": row["image_path"],
                    "image_shape": list(image.size),
                    "instruction": instruction,
                    "prompt": prompt,
                },
                "input_ids": input_ids,
                "tokens": tokenizer.convert_ids_to_tokens(input_ids) if tokenizer is not None else [],
                "input_ids_shape": list(inputs["input_ids"].shape),
                "attention_mask_shape": list(inputs["attention_mask"].shape),
                "pixel_values_shape": list(inputs["pixel_values"].shape),
                "hidden_state_count": len(out.hidden_states),
                "hidden_state_shapes": [list(x.shape) for x in out.hidden_states],
                "projector_features_shape": list(projector.shape),
                "selected_hidden_shape": list(hidden.shape),
                "image_token_range": [image_start, image_end],
                "instruction_positions": instruction_positions,
                "image_hidden_features_shape": list(image_features.shape),
                "instruction_hidden_features_shape": list(instruction_features.shape),
                "hidden_dim": int(hidden.shape[-1]),
                "image_token_count": int(image_features.shape[1]),
                "instruction_token_count": int(instruction_features.shape[1]),
                "action_token_policy": "No action target, labels, or generation call was used; prompt ended at Out:.",
                "openvla_total_parameters": int(sum(p.numel() for p in model.parameters())),
                "openvla_trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "openvla_gradients_all_none_after_forward": grads_none,
                "representative_parameter": first_name,
                "representative_parameter_sha256_before": before_hash,
                "representative_parameter_sha256_after_forward": after_hash,
                "representative_parameter_sha256_match": before_hash == after_hash,
            }
        )
        frozen = {
            "status": "ok",
            "openvla_total_parameters": report["openvla_total_parameters"],
            "openvla_trainable_parameters": report["openvla_trainable_parameters"],
            "representative_parameter": first_name,
            "representative_parameter_sha256": before_hash,
            "gradients_all_none_after_forward": grads_none,
            "optimizer_contains_openvla_parameters": False,
        }
        write_json(report_dir / "frozen_model_audit_before.json", frozen)
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
    write_json(report_dir / "openvla_feature_inspection.json", report)
    md = ["# OpenVLA Feature Inspection", "", f"- Status: `{report['status']}`"]
    for key in [
        "torch",
        "transformers",
        "device",
        "gpu_name",
        "max_vram_bytes",
        "model_class",
        "processor_class",
        "hidden_state_count",
        "selected_hidden_shape",
        "image_hidden_features_shape",
        "instruction_hidden_features_shape",
        "openvla_trainable_parameters",
    ]:
        if key in report:
            md.append(f"- {key}: `{report[key]}`")
    if "traceback" in report:
        md.extend(["", "```", report["traceback"], "```"])
    (report_dir / "openvla_feature_inspection.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"status": report["status"], "report": str(report_dir / "openvla_feature_inspection.json")}, sort_keys=True))
    if report["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
