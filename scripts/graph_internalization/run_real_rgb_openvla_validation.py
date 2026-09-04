from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

from graph_internalization.config import (
    CONDITION_SPECS,
    GRAPH_TEACHER_SHA256,
    LOCKED_MANIFESTS,
    RELATION_VOCAB_SHA256,
    REPORTS,
    ROOT,
    assert_locked_graph_teacher_files,
    load_lora_spec,
    manifest_checksum,
    read_json,
    write_json,
)
from graph_internalization.graph_auxiliary import GraphAuxiliaryModule
from graph_internalization.graph_teacher_loader import load_depth_free_graph_teacher

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


MODEL_PATH = Path(os.environ.get("OPENVLA_BASE_PATH", str(ROOT / "checkpoints/openvla_7b_base")))
DATA_ROOT = ROOT / "data/modified_libero_rlds"
DATASET_NAME = "libero_spatial_no_noops"
EPISODE_MAP = ROOT / "artifacts/openvla_graph_internalization_bundle_v2/rlds_episode_map_libero_spatial_no_noops.json"
RUN_ROOT = ROOT / "runs/real_rgb_openvla_validation"
TINY_MANIFEST = ROOT / "manifests/depth_free_teacher_rgb_tiny_seed101.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def register_openvla() -> None:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)


def maybe_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def tensor_stats_delta(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    delta = (a.detach().float().cpu() - b.detach().float().cpu()).abs()
    return {"max_abs": float(delta.max().item()), "l2": float(torch.linalg.vector_norm(delta).item())}


def grad_norm(params: list[tuple[str, torch.nn.Parameter]]) -> float:
    vals = []
    for _, param in params:
        if param.grad is not None:
            vals.append(torch.linalg.vector_norm(param.grad.detach().float(), 2))
    if not vals:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(vals), 2).item())


def param_delta_norm(before: dict[str, torch.Tensor], params: list[tuple[str, torch.nn.Parameter]]) -> float:
    vals = []
    for name, param in params:
        if name in before:
            vals.append(torch.linalg.vector_norm(param.detach().float().cpu() - before[name].float(), 2))
    if not vals:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(vals), 2).item())


def param_max_delta(before: dict[str, torch.Tensor], params: list[tuple[str, torch.nn.Parameter]]) -> float:
    max_delta = 0.0
    for name, param in params:
        if name in before:
            max_delta = max(max_delta, float((param.detach().cpu() - before[name]).abs().max().item()))
    return max_delta


def clone_params(params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in params}


def restore_params(params: list[tuple[str, torch.nn.Parameter]], snapshot: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in params:
            if name in snapshot:
                param.copy_(snapshot[name].to(device=param.device, dtype=param.dtype))


def lora_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if "lora_" in name and param.requires_grad]


def trainable_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def frozen_base_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if not param.requires_grad and "lora_" not in name]


def base_probe_params(model: torch.nn.Module, limit: int = 8, max_numel: int = 2_000_000) -> list[tuple[str, torch.nn.Parameter]]:
    probes = []
    for name, param in frozen_base_params(model):
        if param.dtype.is_floating_point and param.numel() <= max_numel:
            probes.append((name, param))
        if len(probes) >= limit:
            break
    return probes


def lora_module_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    grouped: dict[str, list[torch.Tensor]] = {}
    for name, param in lora_params(model):
        if param.grad is None:
            continue
        module = name.rsplit(".", 2)[0]
        grouped.setdefault(module, []).append(torch.linalg.vector_norm(param.grad.detach().float(), 2))
    return {
        module: float(torch.linalg.vector_norm(torch.stack(vals), 2).item())
        for module, vals in sorted(grouped.items())
        if vals
    }


def move_batch(batch: dict[str, Any], device: torch.device, include_graph: bool) -> dict[str, Any]:
    out = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "pixel_values": batch["pixel_values"].to(device=device, dtype=torch.bfloat16),
        "labels": batch["labels"].to(device),
    }
    if include_graph:
        out["graph_targets"] = {
            key: value.to(device)
            for key, value in batch["graph_targets"].items()
        }
    return out


def finite_float(value: torch.Tensor) -> float:
    val = float(value.detach().float().cpu().item())
    if not math.isfinite(val):
        raise FloatingPointError(f"Non-finite loss: {val}")
    return val


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def docker_image_info() -> str:
    return os.environ.get(
        "HRI2027_DOCKER_IMAGE",
        "hri2027-libero-eval:latest@sha256:bf01888f7b5f1e6094cd5a6a90ba0e176a76f606f97404ae4d7d557d9e979ec6",
    )


def ensure_tiny_manifest() -> Path:
    if TINY_MANIFEST.exists():
        return TINY_MANIFEST
    base = read_json(LOCKED_MANIFESTS[101])
    selected: list[int] = []
    tasks: dict[str, Any] = {}
    for task_id in ["0", "1"]:
        task = deepcopy(base["tasks"][task_id])
        task["selected_global_episode_indices"] = task["selected_global_episode_indices"][:2]
        task["selected_demo_ids"] = task["selected_demo_ids"][:2]
        task["selected_count"] = 2
        selected.extend(int(x) for x in task["selected_global_episode_indices"])
        tasks[task_id] = task

    rows_by_episode: dict[int, int] = {idx: 0 for idx in selected}
    for line in (ROOT / "artifacts/openvla_graph_internalization_bundle_v2/index.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        idx = int(row["global_episode_index"])
        if idx in rows_by_episode:
            rows_by_episode[idx] += 1

    payload = {
        **{k: v for k, v in base.items() if k not in {"tasks", "selected_global_episode_indices"}},
        "selection_policy": "rgb tiny overfit subset: first 2 demos from tasks 0 and 1 of seed101 holdout",
        "selection_unit": "demonstration",
        "seed": 101,
        "selected_global_episode_indices": selected,
        "selected_demo_count": len(selected),
        "sample_count": sum(rows_by_episode.values()),
        "selected_steps": sum(rows_by_episode.values()),
        "tasks": tasks,
    }
    import hashlib

    checksum_payload = json.dumps(
        {k: v for k, v in payload.items() if k != "checksum"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["checksum"] = hashlib.sha256(checksum_payload).hexdigest()
    write_json(TINY_MANIFEST, payload)
    return TINY_MANIFEST


def set_rlds_env(condition: str, manifest: Path, hook_enabled: bool) -> None:
    for key in [
        "OPENVLA_GRAPH_INTERNALIZATION",
        "OPENVLA_GRAPH_CONDITION",
        "OPENVLA_GRAPH_MANIFEST_JSON",
        "OPENVLA_RLDS_DEMO_SPLIT_JSON",
        "OPENVLA_RLDS_EPISODE_MAP_JSON",
        "OPENVLA_GRAPH_BUNDLE_INDEX",
    ]:
        os.environ.pop(key, None)
    os.environ["OPENVLA_RLDS_DEMO_SPLIT_JSON"] = str(manifest)
    os.environ["OPENVLA_RLDS_EPISODE_MAP_JSON"] = str(EPISODE_MAP)
    if hook_enabled:
        os.environ["OPENVLA_GRAPH_INTERNALIZATION"] = "1"
        os.environ["OPENVLA_GRAPH_CONDITION"] = condition
        os.environ["OPENVLA_GRAPH_MANIFEST_JSON"] = str(manifest)
        os.environ["OPENVLA_GRAPH_BUNDLE_INDEX"] = str(ROOT / "artifacts/openvla_graph_internalization_bundle_v2/index.jsonl")


@dataclass
class Runtime:
    processor: Any
    action_tokenizer: ActionTokenizer
    model: torch.nn.Module
    device: torch.device


def _load_base_model() -> torch.nn.Module:
    return OpenVLAForActionPrediction.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )


def _model_config(model: torch.nn.Module) -> Any:
    return getattr(model, "config", None) or model.get_base_model().config


def load_runtime(seed: int, adapter_path: Path | None = None) -> Runtime:
    set_seed(seed)
    register_openvla()
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    base_model = _load_base_model()
    if adapter_path is None:
        lora_spec = load_lora_spec()
        lora_config = LoraConfig(
            r=int(lora_spec["lora"]["rank"]),
            lora_alpha=int(lora_spec["lora"]["alpha"]),
            lora_dropout=float(lora_spec["lora"]["dropout"]),
            target_modules=str(lora_spec["lora"]["target_modules"]),
            init_lora_weights=str(lora_spec["lora"]["init_lora_weights"]),
        )
        model = get_peft_model(base_model, lora_config)
    else:
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Real OpenVLA validation requires CUDA.")
    model.to(device)
    return Runtime(processor=processor, action_tokenizer=ActionTokenizer(processor.tokenizer), model=model, device=device)


def unload_runtime(runtime: Runtime | None) -> None:
    if runtime is not None:
        del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_loader(
    runtime: Runtime,
    condition: str,
    manifest: Path,
    hook_enabled: bool,
    batch_size: int,
    image_aug: bool = False,
    shuffle_buffer_size: int = 1,
) -> DataLoader:
    set_rlds_env(condition, manifest, hook_enabled)
    batch_transform = RLDSBatchTransform(
        runtime.action_tokenizer,
        runtime.processor.tokenizer,
        image_transform=runtime.processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    dataset = RLDSDataset(
        DATA_ROOT,
        DATASET_NAME,
        batch_transform,
        resize_resolution=tuple(_model_config(runtime.model).image_sizes),
        shuffle_buffer_size=shuffle_buffer_size,
        image_aug=image_aug,
    )
    collator = PaddedCollatorForActionPrediction(
        runtime.processor.tokenizer.model_max_length,
        runtime.processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=0)


def forward_losses(
    runtime: Runtime,
    batch: dict[str, Any],
    condition: str,
    graph_aux: GraphAuxiliaryModule | None,
    action_scale: float = 1.0,
    lambda_graph: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float], Any]:
    spec = CONDITION_SPECS[condition]
    include_graph = spec["uses_graph_aux"]
    uses_action_loss = spec.get("uses_action_loss", True)
    moved = move_batch(batch, runtime.device, include_graph)
    kwargs = {
        "input_ids": moved["input_ids"],
        "attention_mask": moved["attention_mask"],
        "pixel_values": moved["pixel_values"],
        "labels": moved["labels"],
        "output_hidden_states": include_graph,
        "output_projector_features": include_graph,
        "return_dict": True,
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime.model(**kwargs)
        action_loss = output.loss
        if action_loss is None:
            raise ValueError("OpenVLA output.loss is required to preserve the locked token layout.")
        effective_action_scale = action_scale if uses_action_loss else 0.0
        total = action_loss * effective_action_scale if uses_action_loss else None
        metrics = {
            "action_loss": finite_float(action_loss),
            "action_loss_scale": float(effective_action_scale),
            "graph_loss": 0.0,
            "total_loss": 0.0,
        }
        if include_graph:
            assert graph_aux is not None
            graph_losses = graph_aux(output, moved["graph_targets"])
            graph_loss = graph_losses["loss_graph_total"]
            weighted = graph_loss * lambda_graph
            total = total + weighted if total is not None else weighted
            metrics.update(
                {
                    "graph_loss": finite_float(graph_loss),
                    "graph_loss_weighted": finite_float(weighted),
                    "total_loss": finite_float(total),
                }
            )
        elif total is not None:
            metrics["total_loss"] = finite_float(total)
        else:
            raise ValueError(f"{condition} disables action loss but has no graph auxiliary loss.")
    return total, metrics, output


def compare_batches(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in ["input_ids", "attention_mask", "pixel_values", "labels"]:
        av = a[key]
        bv = b[key]
        if av.dtype.is_floating_point:
            max_abs = float((av.float() - bv.float()).abs().max().item())
            out[key] = {"equal": bool(torch.allclose(av.float(), bv.float(), atol=0, rtol=0)), "max_abs": max_abs}
        else:
            out[key] = {"equal": bool(torch.equal(av, bv)), "max_abs": 0.0 if torch.equal(av, bv) else None}
    return out


def run_official_equivalence(seed: int, batch_size: int) -> dict[str, Any]:
    runtime = load_runtime(seed)
    try:
        manifest = LOCKED_MANIFESTS[101]
        official_loader = build_loader(runtime, "rgb_action", manifest, hook_enabled=False, batch_size=batch_size)
        hooked_loader = build_loader(runtime, "rgb_action", manifest, hook_enabled=True, batch_size=batch_size)
        official_batch = next(iter(official_loader))
        hooked_batch = next(iter(hooked_loader))
        batch_cmp = compare_batches(official_batch, hooked_batch)

        runtime.model.eval()
        with torch.no_grad():
            _, official_metrics, official_output = forward_losses(runtime, official_batch, "rgb_action", None)
            _, hooked_metrics, hooked_output = forward_losses(runtime, hooked_batch, "rgb_action", None)
        logits_delta = tensor_stats_delta(official_output.logits, hooked_output.logits)
        action_loss_abs_diff = abs(official_metrics["action_loss"] - hooked_metrics["action_loss"])

        runtime.model.train()
        lora = lora_params(runtime.model)
        trainable = trainable_params(runtime.model)
        base_probes = base_probe_params(runtime.model)
        lora_initial = clone_params(lora)
        base_initial = clone_params(base_probes)

        optimizer = AdamW([param for _, param in trainable], lr=float(load_lora_spec()["training"]["learning_rate"]))
        optimizer.zero_grad(set_to_none=True)
        set_seed(seed)
        official_loss, _, _ = forward_losses(runtime, official_batch, "rgb_action", None)
        official_loss.backward()
        optimizer.step()
        official_lora_delta = param_delta_norm(lora_initial, lora)
        official_base_delta = param_max_delta(base_initial, base_probes)

        restore_params(lora, lora_initial)
        optimizer = AdamW([param for _, param in trainable], lr=float(load_lora_spec()["training"]["learning_rate"]))
        optimizer.zero_grad(set_to_none=True)
        set_seed(seed)
        hooked_loss, _, _ = forward_losses(runtime, hooked_batch, "rgb_action", None)
        hooked_loss.backward()
        optimizer.step()
        hooked_lora_delta = param_delta_norm(lora_initial, lora)
        hooked_base_delta = param_max_delta(base_initial, base_probes)

        trainable_names = [name for name, _ in trainable]
        result = {
            "status": "pass",
            "seed": seed,
            "batch_size": batch_size,
            "manifest": str(manifest),
            "batch_comparison": batch_cmp,
            "logits_delta": logits_delta,
            "official_action_loss": official_metrics["action_loss"],
            "hooked_action_loss": hooked_metrics["action_loss"],
            "action_loss_abs_diff": action_loss_abs_diff,
            "trainable_parameter_count": sum(param.numel() for _, param in trainable),
            "trainable_parameter_names": trainable_names,
            "optimizer": {
                "type": "AdamW",
                "parameter_group_count": 1,
                "parameter_count": sum(param.numel() for _, param in trainable),
                "lr": float(load_lora_spec()["training"]["learning_rate"]),
            },
            "one_step_parameter_delta": {
                "official_lora_delta_l2": official_lora_delta,
                "hooked_lora_delta_l2": hooked_lora_delta,
                "lora_delta_abs_diff": abs(official_lora_delta - hooked_lora_delta),
                "official_base_probe_max_abs_delta": official_base_delta,
                "hooked_base_probe_max_abs_delta": hooked_base_delta,
                "base_probe_names": [name for name, _ in base_probes],
            },
        }
        if not all(v["equal"] for v in batch_cmp.values()):
            result["status"] = "fail"
        if logits_delta["max_abs"] > 1e-5 or action_loss_abs_diff > 1e-6:
            result["status"] = "fail"
        if official_lora_delta <= 0 or hooked_lora_delta <= 0 or official_base_delta != 0 or hooked_base_delta != 0:
            result["status"] = "fail"
        return result
    finally:
        unload_runtime(runtime)


def run_gradient_diagnostics(seed: int, batch_size: int, lambda_graph: float, conditions: list[str]) -> dict[str, Any]:
    runtime = load_runtime(seed)
    graph_aux = GraphAuxiliaryModule(load_depth_free_graph_teacher(device=str(runtime.device)), lambda_graph=lambda_graph).to(runtime.device)
    try:
        manifest = LOCKED_MANIFESTS[101]
        result: dict[str, Any] = {"status": "pass", "seed": seed, "batch_size": batch_size, "lambda_graph": lambda_graph, "conditions": {}}
        for condition in conditions:
            loader = build_loader(runtime, condition, manifest, hook_enabled=True, batch_size=batch_size)
            batch = next(iter(loader))
            runtime.model.train()
            runtime.model.zero_grad(set_to_none=True)
            graph_aux.zero_grad(set_to_none=True)
            lora = lora_params(runtime.model)
            trainable = trainable_params(runtime.model)
            frozen = frozen_base_params(runtime.model)
            base_probes = base_probe_params(runtime.model)
            lora_before = clone_params(lora)
            base_before = clone_params(base_probes)
            teacher_before = clone_params(list(graph_aux.teacher.named_parameters()))
            optimizer = AdamW([param for _, param in trainable], lr=float(load_lora_spec()["training"]["learning_rate"]))
            spec = CONDITION_SPECS[condition]
            loss, metrics, _ = forward_losses(
                runtime,
                batch,
                condition,
                graph_aux if spec["uses_graph_aux"] else None,
                action_scale=1.0,
                lambda_graph=lambda_graph,
            )
            loss.backward()
            lora_grad = grad_norm(lora)
            base_grad = grad_norm(frozen)
            teacher_grad = grad_norm(list(graph_aux.teacher.named_parameters()))
            module_grads = lora_module_grad_norms(runtime.model)
            optimizer.step()
            lora_delta = param_delta_norm(lora_before, lora)
            base_delta = param_max_delta(base_before, base_probes)
            teacher_delta = param_max_delta(teacher_before, list(graph_aux.teacher.named_parameters()))
            condition_status = "pass"
            if condition == "rgb_action" and metrics["action_loss"] <= 0:
                condition_status = "fail"
            if spec["uses_graph_aux"] and metrics["graph_loss"] <= 0:
                condition_status = "fail"
            if lora_grad <= 0 or lora_delta <= 0 or base_grad != 0 or base_delta != 0 or teacher_grad != 0 or teacher_delta != 0:
                condition_status = "fail"
            if spec["uses_graph_aux"] and not any(v > 0 for v in module_grads.values()):
                condition_status = "fail"
            result["conditions"][condition] = {
                "status": condition_status,
                **metrics,
                "lora_grad_norm": lora_grad,
                "base_grad_norm": base_grad,
                "graph_teacher_grad_norm": teacher_grad,
                "lora_parameter_delta_l2": lora_delta,
                "base_probe_max_abs_delta": base_delta,
                "graph_teacher_max_abs_delta": teacher_delta,
                "lora_module_grad_norms": module_grads,
                "sample_keys": batch.get("sample_key", []),
            }
            if condition_status != "pass":
                result["status"] = "fail"
            restore_params(lora, lora_before)
        return result
    finally:
        unload_runtime(runtime)


def save_checkpoint(
    path: Path,
    runtime: Runtime,
    optimizer: AdamW,
    scheduler: LambdaLR,
    condition: str,
    global_step: int,
    manifest: Path,
    lambda_graph: float,
    micro_step: int = 0,
    training_seed: int | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    runtime.model.save_pretrained(path / "lora_adapter")
    runtime.processor.save_pretrained(path / "processor")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "micro_step": micro_step,
            "training_seed": training_seed,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "condition": condition,
            "manifest": str(manifest),
            "manifest_checksum": manifest_checksum(manifest),
            "graph_teacher_checkpoint_sha256": GRAPH_TEACHER_SHA256,
            "relation_vocabulary_sha256": RELATION_VOCAB_SHA256,
            "lambda_graph": lambda_graph,
            "git_commit": git_commit(),
            "docker_image": docker_image_info(),
        },
        path / "training_state.pt",
    )
    write_json(
        path / "checkpoint_metadata.json",
        {
            "global_step": global_step,
            "micro_step": micro_step,
            "training_seed": training_seed,
            "condition": condition,
            "manifest": str(manifest),
            "manifest_checksum": manifest_checksum(manifest),
            "graph_teacher_checkpoint_sha256": GRAPH_TEACHER_SHA256,
            "relation_vocabulary_sha256": RELATION_VOCAB_SHA256,
            "lambda_graph": lambda_graph,
            "git_commit": git_commit(),
            "docker_image": docker_image_info(),
        },
    )


def load_checkpoint(path: Path, seed: int, lr: float) -> tuple[Runtime, AdamW, LambdaLR, dict[str, Any]]:
    runtime = load_runtime(seed, adapter_path=path / "lora_adapter")
    optimizer = AdamW([param for _, param in trainable_params(runtime.model)], lr=lr)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    return runtime, optimizer, scheduler, state


def run_tiny_overfit(
    seed: int,
    updates: int,
    batch_size: int,
    lambda_graph: float,
    conditions: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    tiny_manifest = ensure_tiny_manifest()
    lora_spec = load_lora_spec()
    lr = float(lora_spec["training"]["learning_rate"])
    summary: dict[str, Any] = {
        "status": "pass",
        "seed": seed,
        "updates": updates,
        "batch_size": batch_size,
        "lambda_graph": lambda_graph,
        "manifest": str(tiny_manifest),
        "manifest_checksum": manifest_checksum(tiny_manifest),
        "conditions": {},
    }
    resume_report: dict[str, Any] = {"status": "pass", "conditions": {}}

    for condition in conditions:
        runtime = load_runtime(seed)
        graph_aux = GraphAuxiliaryModule(load_depth_free_graph_teacher(device=str(runtime.device)), lambda_graph=lambda_graph).to(runtime.device)
        optimizer = AdamW([param for _, param in trainable_params(runtime.model)], lr=lr)
        scheduler = LambdaLR(optimizer, lambda _: 1.0)
        loader = build_loader(runtime, condition, tiny_manifest, hook_enabled=True, batch_size=batch_size)
        iterator = iter(loader)
        lora = lora_params(runtime.model)
        base_probes = base_probe_params(runtime.model)
        lora_initial = clone_params(lora)
        base_initial = clone_params(base_probes)
        teacher_initial = clone_params(list(graph_aux.teacher.named_parameters()))
        metrics = []
        checkpoint_step = max(2, updates // 2)
        checkpoint_dir = RUN_ROOT / condition / f"checkpoint_step_{checkpoint_step}"
        resumed = False
        loss_at_checkpoint = None
        loss_after_resume = None
        global_step = 0
        try:
            while global_step < updates:
                start = time.perf_counter()
                batch = next(iterator)
                runtime.model.train()
                optimizer.zero_grad(set_to_none=True)
                graph_aux.zero_grad(set_to_none=True)
                loss, loss_metrics, _ = forward_losses(
                    runtime,
                    batch,
                    condition,
                    graph_aux if CONDITION_SPECS[condition]["uses_graph_aux"] else None,
                    action_scale=1.0,
                    lambda_graph=lambda_graph,
                )
                loss.backward()
                lora_grad = grad_norm(lora_params(runtime.model))
                optimizer.step()
                scheduler.step()
                global_step += 1
                if torch.cuda.is_available():
                    memory_allocated = int(torch.cuda.max_memory_allocated(runtime.device))
                    torch.cuda.reset_peak_memory_stats(runtime.device)
                else:
                    memory_allocated = 0
                row = {
                    "step": global_step,
                    **loss_metrics,
                    "lora_grad_norm": lora_grad,
                    "lora_parameter_delta_l2": param_delta_norm(lora_initial, lora_params(runtime.model)),
                    "base_probe_max_abs_delta": param_max_delta(base_initial, base_probes),
                    "graph_teacher_max_abs_delta": param_max_delta(teacher_initial, list(graph_aux.teacher.named_parameters())),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "sample_keys": batch.get("sample_key", []),
                    "has_nan_or_inf": not all(math.isfinite(float(v)) for k, v in loss_metrics.items() if isinstance(v, float)),
                    "gpu_memory_allocated_bytes": memory_allocated,
                    "iteration_seconds": time.perf_counter() - start,
                }
                metrics.append(row)
                if global_step == checkpoint_step:
                    loss_at_checkpoint = row["total_loss"]
                    save_checkpoint(checkpoint_dir, runtime, optimizer, scheduler, condition, global_step, tiny_manifest, lambda_graph)
                    unload_runtime(runtime)
                    runtime, optimizer, scheduler, state = load_checkpoint(checkpoint_dir, seed, lr)
                    if CONDITION_SPECS[condition]["uses_graph_aux"]:
                        assert_locked_graph_teacher_files()
                    loader = build_loader(runtime, condition, tiny_manifest, hook_enabled=True, batch_size=batch_size)
                    iterator = iter(loader)
                    graph_aux = GraphAuxiliaryModule(load_depth_free_graph_teacher(device=str(runtime.device)), lambda_graph=lambda_graph).to(runtime.device)
                    lora = lora_params(runtime.model)
                    base_probes = base_probe_params(runtime.model)
                    global_step = int(state["global_step"])
                    resumed = True
                elif resumed and loss_after_resume is None:
                    loss_after_resume = row["total_loss"]
        finally:
            final_checkpoint = RUN_ROOT / condition / f"checkpoint_step_{updates}"
            save_checkpoint(final_checkpoint, runtime, optimizer, scheduler, condition, global_step, tiny_manifest, lambda_graph)
            unload_runtime(runtime)

        first = metrics[0]
        last = metrics[-1]
        action_loss_decreased = last["action_loss"] < first["action_loss"]
        graph_ok = True
        if CONDITION_SPECS[condition]["uses_graph_aux"]:
            graph_ok = math.isfinite(last["graph_loss"]) and last["graph_loss"] <= first["graph_loss"] * 1.25
        condition_status = "pass"
        if condition == "rgb_action" and not action_loss_decreased:
            condition_status = "fail"
        if CONDITION_SPECS[condition]["uses_graph_aux"] and (not math.isfinite(last["action_loss"]) or not graph_ok):
            condition_status = "fail"
        if last["lora_parameter_delta_l2"] <= 0 or last["base_probe_max_abs_delta"] != 0 or last["graph_teacher_max_abs_delta"] != 0:
            condition_status = "fail"
        if not resumed or loss_after_resume is None or not math.isfinite(float(loss_after_resume)):
            condition_status = "fail"

        summary["conditions"][condition] = {
            "status": condition_status,
            "first": first,
            "last": last,
            "action_loss_decreased": action_loss_decreased,
            "graph_loss_stable_or_decreased": graph_ok,
            "checkpoint_step": checkpoint_step,
            "final_checkpoint_step": updates,
            "checkpoint_dir": str(checkpoint_dir),
            "final_checkpoint_dir": str(RUN_ROOT / condition / f"checkpoint_step_{updates}"),
            "metrics": metrics,
        }
        resume_report["conditions"][condition] = {
            "status": "pass" if condition_status == "pass" else "fail",
            "checkpoint_dir": str(checkpoint_dir),
            "global_step_before_resume": checkpoint_step,
            "global_step_after_resume": checkpoint_step + 1 if resumed else None,
            "loss_at_checkpoint": loss_at_checkpoint,
            "loss_after_resume": loss_after_resume,
            "loss_after_resume_finite": loss_after_resume is not None and math.isfinite(float(loss_after_resume)),
            "lora_state_reloaded": resumed,
            "optimizer_scheduler_reloaded": resumed,
            "graph_teacher_strict_load": CONDITION_SPECS[condition]["uses_graph_aux"],
        }
        if condition_status != "pass":
            summary["status"] = "fail"
            resume_report["status"] = "fail"
    return summary, resume_report


def write_markdown(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [f"# {title}", "", f"- status: `{payload.get('status')}`"]
    for key in ["seed", "batch_size", "updates", "lambda_graph", "manifest", "manifest_checksum"]:
        if key in payload:
            lines.append(f"- {key}: `{payload[key]}`")
    if "conditions" in payload:
        lines.append("")
        lines.append("## Conditions")
        for condition, row in payload["conditions"].items():
            lines.append(f"- `{condition}`: `{row.get('status')}`")
            if "first" in row and "last" in row:
                lines.append(
                    f"  - action_loss: `{row['first']['action_loss']}` -> `{row['last']['action_loss']}`"
                )
                lines.append(
                    f"  - graph_loss: `{row['first'].get('graph_loss', 0.0)}` -> `{row['last'].get('graph_loss', 0.0)}`"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tiny-updates", type=int, default=50)
    parser.add_argument("--lambda-graph", type=float, default=0.1)
    parser.add_argument("--conditions", nargs="+", default=["rgb_action", "rgb_graph"])
    args = parser.parse_args()
    for condition in args.conditions:
        if condition not in CONDITION_SPECS:
            raise SystemExit(f"Unknown condition: {condition}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    equivalence = run_official_equivalence(args.seed, args.batch_size)
    write_json(REPORTS / "real_rgb_action_official_equivalence.json", equivalence)
    write_markdown(REPORTS / "real_rgb_action_official_equivalence.md", "Real RGB Action Official Equivalence", equivalence)

    gradients = run_gradient_diagnostics(args.seed, args.batch_size, args.lambda_graph, args.conditions)
    write_json(REPORTS / "real_rgb_lora_gradient_diagnostics.json", gradients)
    write_markdown(REPORTS / "real_rgb_lora_gradient_diagnostics.md", "Real RGB LoRA Gradient Diagnostics", gradients)

    tiny, resume = run_tiny_overfit(args.seed, args.tiny_updates, args.batch_size, args.lambda_graph, args.conditions)
    write_json(REPORTS / "real_rgb_tiny_overfit_metrics.json", tiny)
    write_markdown(REPORTS / "real_rgb_tiny_overfit_summary.md", "Real RGB Tiny Overfit", tiny)
    write_json(REPORTS / "real_rgb_checkpoint_resume.json", resume)
    write_markdown(REPORTS / "real_rgb_checkpoint_resume.md", "Real RGB Checkpoint Resume", resume)

    status = "pass" if equivalence["status"] == gradients["status"] == tiny["status"] == resume["status"] == "pass" else "fail"
    print(json.dumps({"status": status, "equivalence": equivalence["status"], "gradients": gradients["status"], "tiny": tiny["status"], "resume": resume["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
