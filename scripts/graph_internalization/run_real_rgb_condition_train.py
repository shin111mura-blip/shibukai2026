from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from graph_internalization.config import CONDITION_SPECS, LOCKED_MANIFESTS, load_lora_spec, manifest_checksum, write_json
from graph_internalization.graph_auxiliary import GraphAuxiliaryModule
from graph_internalization.graph_teacher_loader import load_depth_free_graph_teacher

from scripts.graph_internalization.run_real_rgb_openvla_validation import (
    RUN_ROOT,
    base_probe_params,
    build_loader,
    docker_image_info,
    forward_losses,
    git_commit,
    grad_norm,
    load_runtime,
    load_checkpoint,
    lora_params,
    param_delta_norm,
    param_max_delta,
    save_checkpoint,
    trainable_params,
    unload_runtime,
)


def latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = []
    for path in run_dir.glob("checkpoint_step_*"):
        try:
            step = int(path.name.rsplit("_", 1)[1])
        except Exception:
            continue
        if (path / "training_state.pt").exists() and (path / "lora_adapter").exists():
            candidates.append((step, path))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def csv_mode_and_header(path: Path) -> tuple[str, bool]:
    return ("a", False) if path.exists() and path.stat().st_size > 0 else ("w", True)


def finite_metrics(row: dict[str, Any]) -> bool:
    return all(
        math.isfinite(float(row[key]))
        for key in ["action_loss", "graph_loss", "total_loss", "lora_grad_norm"]
    )


def parameter_rows(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, param in model.named_parameters():
        rows.append(
            {
                "name": name,
                "shape": list(param.shape),
                "numel": int(param.numel()),
                "requires_grad": bool(param.requires_grad),
                "module_category": "lora" if "lora_" in name else "base",
            }
        )
    return rows


def write_parameter_reports(run_dir: Path, runtime: Any, graph_aux: GraphAuxiliaryModule | None) -> dict[str, Any]:
    rows = parameter_rows(runtime.model)
    trainable = [row for row in rows if row["requires_grad"]]
    frozen = [row for row in rows if not row["requires_grad"]]
    (run_dir / "trainable_parameters.txt").write_text(
        "\n".join(f"{row['name']}\t{row['shape']}\t{row['numel']}\t{row['module_category']}" for row in trainable) + "\n",
        encoding="utf-8",
    )
    (run_dir / "frozen_parameters.txt").write_text(
        "\n".join(f"{row['name']}\t{row['shape']}\t{row['numel']}\t{row['module_category']}" for row in frozen) + "\n",
        encoding="utf-8",
    )
    graph_teacher_rows = []
    if graph_aux is not None:
        graph_teacher_rows = [
            {
                "name": name,
                "shape": list(param.shape),
                "numel": int(param.numel()),
                "requires_grad": bool(param.requires_grad),
                "grad_is_none": param.grad is None,
            }
            for name, param in graph_aux.teacher.named_parameters()
        ]
    summary = {
        "trainable_parameter_count": int(sum(row["numel"] for row in trainable)),
        "trainable_parameter_names": [row["name"] for row in trainable],
        "frozen_parameter_count": int(sum(row["numel"] for row in frozen)),
        "frozen_parameter_names": [row["name"] for row in frozen],
        "parameter_count_total": int(sum(row["numel"] for row in rows)),
        "graph_teacher_parameter_count": int(sum(row["numel"] for row in graph_teacher_rows)),
        "graph_teacher_all_requires_grad_false": all(not row["requires_grad"] for row in graph_teacher_rows),
        "graph_teacher_all_grad_none": all(row["grad_is_none"] for row in graph_teacher_rows),
        "graph_teacher_parameter_names": [row["name"] for row in graph_teacher_rows],
    }
    write_json(
        run_dir / "parameter_summary.json",
        {
            **summary,
            "parameters": rows,
            "graph_teacher_parameters": graph_teacher_rows,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Real OpenVLA RGB-only condition trainer.")
    parser.add_argument(
        "--condition",
        choices=["rgb_action", "rgb_graph", "rgb_graph_no_action"],
        required=True,
    )
    parser.add_argument("--seed", type=int, choices=[101, 202, 303], required=True)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shuffle-buffer-size", type=int, default=100000)
    parser.add_argument("--lambda-graph", type=float, default=0.1)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint directory, or 'latest'.")
    args = parser.parse_args()

    manifest = LOCKED_MANIFESTS[args.seed]
    lora_spec = load_lora_spec()
    lr = float(lora_spec["training"]["learning_rate"])
    grad_accumulation_steps = int(
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else lora_spec["training"]["gradient_accumulation_steps"]
    )
    if args.batch_size != int(lora_spec["training"]["batch_size_per_device"]):
        raise SystemExit(
            f"batch size {args.batch_size} does not match locked per-device batch size "
            f"{lora_spec['training']['batch_size_per_device']}"
        )
    if grad_accumulation_steps != int(lora_spec["training"]["gradient_accumulation_steps"]):
        raise SystemExit(
            f"gradient_accumulation_steps {grad_accumulation_steps} does not match locked value "
            f"{lora_spec['training']['gradient_accumulation_steps']}"
        )
    run_dir = args.run_root / args.condition / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_path: Path | None = None
    if args.resume_from == "latest":
        resume_path = latest_checkpoint(run_dir)
    elif args.resume_from:
        resume_path = Path(args.resume_from)

    if resume_path is not None:
        runtime, optimizer, scheduler, state = load_checkpoint(resume_path, args.training_seed, lr)
        global_step = int(state["global_step"])
        micro_step = int(state.get("micro_step", global_step * grad_accumulation_steps))
        if state.get("condition") != args.condition:
            raise SystemExit(f"checkpoint condition {state.get('condition')!r} does not match {args.condition!r}")
        if float(state.get("lambda_graph", args.lambda_graph)) != float(args.lambda_graph):
            raise SystemExit("checkpoint lambda_graph does not match requested lambda_graph")
    else:
        runtime = load_runtime(args.training_seed)
        optimizer = AdamW([param for _, param in trainable_params(runtime.model)], lr=lr)
        scheduler = LambdaLR(optimizer, lambda _: 1.0)
        global_step = 0
        micro_step = 0

    condition_spec = CONDITION_SPECS[args.condition]
    graph_aux = (
        GraphAuxiliaryModule(load_depth_free_graph_teacher(device=str(runtime.device)), lambda_graph=args.lambda_graph).to(runtime.device)
        if condition_spec["uses_graph_aux"]
        else None
    )
    parameter_summary = write_parameter_reports(run_dir, runtime, graph_aux)
    loader = build_loader(
        runtime,
        args.condition,
        manifest,
        hook_enabled=True,
        batch_size=args.batch_size,
        image_aug=bool(lora_spec["training"]["image_aug"]),
        shuffle_buffer_size=args.shuffle_buffer_size,
    )
    iterator = iter(loader)
    lora_initial = {name: param.detach().cpu().clone() for name, param in lora_params(runtime.model)}
    base_probes = base_probe_params(runtime.model)
    base_initial = {name: param.detach().cpu().clone() for name, param in base_probes}
    teacher_initial = {
        name: param.detach().cpu().clone()
        for name, param in (graph_aux.teacher.named_parameters() if graph_aux is not None else [])
    }

    metrics_path = run_dir / "metrics.csv"
    mode, write_header = csv_mode_and_header(metrics_path)
    with metrics_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_step",
                "micro_step",
                "action_loss",
                "graph_loss",
                "total_loss",
                "accumulated_loss",
                "lora_grad_norm",
                "lora_parameter_delta_l2",
                "base_probe_max_abs_delta",
                "graph_teacher_max_abs_delta",
                "learning_rate",
                "gpu_memory_allocated_bytes",
                "iteration_seconds",
            ],
        )
        if write_header:
            writer.writeheader()
        try:
            optimizer.zero_grad(set_to_none=True)
            if graph_aux is not None:
                graph_aux.zero_grad(set_to_none=True)
            while global_step < args.max_steps:
                accumulated_metrics: list[dict[str, float]] = []
                start = time.perf_counter()
                for _ in range(grad_accumulation_steps):
                    batch = next(iterator)
                    micro_step += 1
                    runtime.model.train()
                    loss, loss_metrics, _ = forward_losses(
                        runtime,
                        batch,
                        args.condition,
                        graph_aux,
                        action_scale=1.0,
                        lambda_graph=args.lambda_graph,
                    )
                    (loss / grad_accumulation_steps).backward()
                    accumulated_metrics.append(loss_metrics)
                lora_grad = grad_norm(lora_params(runtime.model))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if graph_aux is not None:
                    graph_aux.zero_grad(set_to_none=True)
                global_step += 1
                memory_allocated = int(torch.cuda.max_memory_allocated(runtime.device)) if torch.cuda.is_available() else 0
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(runtime.device)
                mean_metrics = {
                    key: sum(float(m[key]) for m in accumulated_metrics) / len(accumulated_metrics)
                    for key in ["action_loss", "graph_loss", "total_loss"]
                }
                row = {
                    "global_step": global_step,
                    "micro_step": micro_step,
                    "action_loss": mean_metrics["action_loss"],
                    "graph_loss": mean_metrics["graph_loss"],
                    "total_loss": mean_metrics["total_loss"],
                    "accumulated_loss": sum(float(m["total_loss"]) for m in accumulated_metrics),
                    "lora_grad_norm": lora_grad,
                    "lora_parameter_delta_l2": param_delta_norm(lora_initial, lora_params(runtime.model)),
                    "base_probe_max_abs_delta": param_max_delta(base_initial, base_probes),
                    "graph_teacher_max_abs_delta": param_max_delta(
                        teacher_initial,
                        list(graph_aux.teacher.named_parameters()) if graph_aux is not None else [],
                    ),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gpu_memory_allocated_bytes": memory_allocated,
                    "iteration_seconds": time.perf_counter() - start,
                }
                if not finite_metrics(row):
                    raise FloatingPointError(f"Non-finite metric at global_step {global_step}: {row}")
                writer.writerow(row)
                f.flush()
                if global_step % args.save_steps == 0 or global_step == args.max_steps:
                    save_checkpoint(
                        run_dir / f"checkpoint_step_{global_step}",
                        runtime,
                        optimizer,
                        scheduler,
                        args.condition,
                        global_step,
                        manifest,
                        args.lambda_graph,
                        micro_step=micro_step,
                        training_seed=args.training_seed,
                    )
        finally:
            unload_runtime(runtime)

    write_json(
        run_dir / "run_metadata.json",
        {
            "status": "done",
            "condition": args.condition,
            "seed": args.seed,
            "manifest": str(manifest),
            "manifest_checksum": manifest_checksum(manifest),
            "max_steps": args.max_steps,
            "save_steps": args.save_steps,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": grad_accumulation_steps,
            "effective_batch_size_per_gpu": args.batch_size * grad_accumulation_steps,
            "lambda_graph": args.lambda_graph,
            "uses_action_loss": condition_spec.get("uses_action_loss", True),
            "uses_graph_aux": condition_spec["uses_graph_aux"],
            "parameter_summary": parameter_summary,
            "global_step": global_step,
            "micro_step": micro_step,
            "resumed_from": str(resume_path) if resume_path else None,
            "git_commit": git_commit(),
            "docker_image": docker_image_info(),
        },
    )
    print(json.dumps({"status": "done", "run_dir": str(run_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
