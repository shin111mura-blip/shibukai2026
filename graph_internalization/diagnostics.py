from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import CONDITION_SPECS, LOCKED_MANIFESTS, REPORTS, assert_locked_graph_teacher_files, write_json
from .graph_teacher_loader import load_depth_free_graph_teacher
from .sidecar_dataset import assert_same_keys_for_conditions, load_bundle_rows, manifest_selection

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def grad_norm(parameters: Iterable[Any]) -> float:
    if torch is None:
        return 0.0
    vals = []
    for param in parameters:
        if getattr(param, "grad", None) is not None:
            vals.append(torch.linalg.vector_norm(param.grad.detach(), 2))
    if not vals:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(vals), 2).item())


def parameter_max_abs_delta(before: dict[str, Any], module: Any) -> float:
    if torch is None:
        return 0.0
    max_delta = 0.0
    for name, param in module.named_parameters():
        delta = (param.detach().cpu() - before[name]).abs().max().item()
        max_delta = max(max_delta, float(delta))
    return max_delta


def snapshot_parameters(module: Any) -> dict[str, Any]:
    return {name: param.detach().cpu().clone() for name, param in module.named_parameters()}


def run_locked_spec_checks() -> dict[str, Any]:
    lock = assert_locked_graph_teacher_files()
    loaded = load_depth_free_graph_teacher(device="cpu")
    teacher_trainable = sum(param.numel() for param in loaded.model.parameters() if param.requires_grad)
    if teacher_trainable != 0:
        raise AssertionError(f"Graph teacher has trainable params: {teacher_trainable}")
    return {**lock, "strict_load": True, "graph_teacher_trainable_params": teacher_trainable}


def run_manifest_checks() -> dict[str, Any]:
    rows = load_bundle_rows()
    out = {}
    for seed, path in LOCKED_MANIFESTS.items():
        if not path.exists():
            continue
        selection = manifest_selection(path, rows)
        out[str(seed)] = {
            **assert_same_keys_for_conditions(selection, CONDITION_SPECS),
            "overlap_with_graph_generator_train": selection.overlap_with_graph_generator_train,
        }
    return out


def write_markdown_report(path: Path, title: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", *rows, ""]) + "\n", encoding="utf-8")


def write_diagnostic_json(path: Path, payload: dict[str, Any]) -> None:
    serializable = json.loads(json.dumps(payload, default=str))
    write_json(path, serializable)
