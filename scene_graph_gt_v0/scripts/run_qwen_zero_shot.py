#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from scene_graph.canonicalize import make_config_hash
from scene_graph.qwen_parser import parse_qwen_graph


PREDICATES = ["left_of", "right_of", "above", "below", "front_of", "behind", "on", "inside", "contains", "grasping"]
DEFAULT_MODELS = ["Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-id", default="demo_0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_gt_v0"))
    parser.add_argument("--model", default=DEFAULT_MODELS[0])
    parser.add_argument("--fallback-model", default=DEFAULT_MODELS[1])
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/scene_graph_gt_v0/hf_cache"))
    parser.add_argument("--frames", default=None, help="Optional comma-separated frame ids to run instead of selected_frames.json.")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--dry-run", action="store_true", help="Write invalid placeholder records without loading Qwen.")
    parser.add_argument("--parse-existing", action="store_true", help="Re-parse existing raw outputs with current per-frame visible node constraints.")
    return parser.parse_args()


def prompt(task_instruction: str, node_ids: list[str]) -> str:
    return (
        "Return only JSON matching this schema: "
        "{\"nodes\":[{\"id\":string,\"category\":string,\"entity_type\":\"object|fixture|gripper\",\"present\":boolean,\"visible\":boolean}],"
        "\"binary_edges\":[{\"subject\":string,\"predicate\":string,\"object\":string}]}.\n"
        f"Task instruction: {task_instruction}\n"
        f"Allowed node ids for this frame: {node_ids}\n"
        f"Allowed predicates: {PREDICATES}\n"
        "Use only the allowed node ids. Do not include task objects that are not in this per-frame list.\n"
        "Definitions: left_of/right_of are image-plane relations only for objects in a similar horizontal band; "
        "above/below require approximate x alignment, not just a y-coordinate difference; "
        "front_of means closer to the camera than the object; behind is the inverse of front_of; "
        "on means the subject is visibly resting on the object's top support surface; "
        "inside means subject is inside container; contains is inverse of inside; "
        "grasping means the gripper fingers visibly enclose/contact the object.\n"
        "Do not output between, touching, near, overlapping, holding, ternary_edges, or unknown node ids. "
        "If a relation is not visible in the image, omit it."
    )


def gpu_memory() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return {"nvidia_smi": out.strip()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def load_qwen(model_name: str, cache_dir: Path):
    import torch
    import transformers
    from transformers import AutoProcessor

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    class_name = "Qwen3VLForConditionalGeneration"
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
        class_name = "AutoModelForMultimodalLM"
    if model_cls is None:
        raise ImportError(
            "Qwen3-VL requires a newer transformers build with "
            "Qwen3VLForConditionalGeneration or AutoModelForMultimodalLM. "
            f"Installed transformers={getattr(transformers, '__version__', 'unknown')}"
        )
    base_kwargs = {"cache_dir": str(cache_dir), "device_map": "auto"}
    attempts = [
        ("flash_attention_2", {"dtype": dtype, "attn_implementation": "flash_attention_2", **base_kwargs}),
        ("sdpa", {"dtype": dtype, "attn_implementation": "sdpa", **base_kwargs}),
        ("sdpa_torch_dtype", {"torch_dtype": dtype, "attn_implementation": "sdpa", **base_kwargs}),
    ]
    last_exc = None
    for attention, kwargs in attempts:
        try:
            model = model_cls.from_pretrained(model_name, **kwargs)
            break
        except TypeError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            last_exc = exc
            if attention == "flash_attention_2":
                continue
            raise
    else:
        raise last_exc or RuntimeError("failed to load Qwen3-VL")
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=str(cache_dir))
    return model, processor, {
        "model_name": model_name,
        "dtype": str(dtype),
        "attention": attention,
        "model_class": class_name,
        "transformers_version": getattr(transformers, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def generate(model, processor, image_path: Path, text: str, max_new_tokens: int) -> str:
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    messages = [{"role": "user", "content": [{"type": "image", "image": str(image_path.resolve())}, {"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "reports" / "selected_frames.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected_frames"]
    if args.frames:
        wanted = {int(item.strip()) for item in args.frames.split(",") if item.strip()}
        selected = [item for item in selected if int(item["frame_id"]) in wanted]
    task_report = json.loads((args.output_dir / "reports" / "investigation.json").read_text(encoding="utf-8"))
    task_instruction = task_report["target_task"]["instruction"]
    cfg = {
        "model": args.model,
        "fallback_model": args.fallback_model,
        "cache_dir": str(args.cache_dir),
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "prompt_version": "qwen_zero_shot_v1_frame_visible_nodes",
        "parser_constraint": "per_frame_rule_observable_nodes",
    }
    config_hash = make_config_hash(cfg)
    status = {"config": cfg, "frames": [], "load": None, "vram_before": gpu_memory(), "vram_after": None}
    model = processor = None
    model_info = {}
    if args.parse_existing:
        status["load"] = {"ok": False, "parse_existing": True}
    elif not args.dry_run:
        try:
            try:
                model, processor, model_info = load_qwen(args.model, args.cache_dir)
            except Exception:
                model, processor, model_info = load_qwen(args.fallback_model, args.cache_dir)
            status["load"] = {"ok": True, **model_info}
        except Exception as exc:
            status["load"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        status["load"] = {"ok": False, "dry_run": True}
    for item in selected:
        frame_id = int(item["frame_id"])
        image_path = args.output_dir / "frames" / args.demo_id / f"{frame_id:06d}.png"
        raw_path = args.output_dir / "qwen_zero_shot" / "raw" / args.demo_id / f"{frame_id:06d}.txt"
        parsed_path = args.output_dir / "qwen_zero_shot" / "parsed" / args.demo_id / f"{frame_id:06d}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        rule_graph = json.loads((args.output_dir / "rule_based" / "observable_graph" / args.demo_id / f"{frame_id:06d}.json").read_text(encoding="utf-8"))
        node_ids = [node["id"] for node in rule_graph["nodes"]]
        if args.parse_existing:
            if raw_path.exists():
                raw = raw_path.read_text(encoding="utf-8")
            else:
                raw = json.dumps({"invalid": True, "reason": "missing_raw_output"}, sort_keys=True)
        elif model is None or processor is None:
            raw = json.dumps({"invalid": True, "reason": status["load"]}, sort_keys=True)
        else:
            raw = generate(model, processor, image_path, prompt(task_instruction, node_ids), args.max_new_tokens)
        if not args.parse_existing:
            raw_path.write_text(raw + "\n", encoding="utf-8")
        parsed, info = parse_qwen_graph(
            raw_text=raw,
            task_id=str(task_report["target_task"]["task_id"]),
            demo_id=args.demo_id,
            frame_id=frame_id,
            allowed_node_ids=node_ids,
            config_hash=config_hash,
        )
        if parsed is not None:
            parsed_path.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif parsed_path.exists():
            parsed_path.unlink()
        status["frames"].append({
            "frame_id": frame_id,
            "raw": str(raw_path),
            "parsed": str(parsed_path) if parsed else None,
            "allowed_node_ids": node_ids,
            **info,
        })
    status["vram_after"] = gpu_memory()
    out = args.output_dir / "reports" / "qwen_zero_shot_status.json"
    out.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
