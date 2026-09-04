#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph_generator.graph_generator.decoding import decode_graph
from scene_graph_generator.graph_generator.feature_extractor import OpenVLAFeatureExtractor
from scene_graph_generator.graph_generator.masks import relation_validity_mask
from scene_graph_generator.graph_generator.metrics import summarize_examples
from scene_graph_generator.graph_generator.metrics_3d import xyz_metrics
from scene_graph_generator.graph_generator.schema import file_sha256, graph_node_ids, graph_triplets, validate_graph, write_json
from scripts.rollout_xyz_targets import RolloutXyzTargetCache


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


def is_hf_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def resolve_checkpoint(path: Path) -> Path:
    if is_hf_checkpoint(path):
        return path
    candidates = []
    for env_name in ("OPENVLA_BASE_CHECKPOINT", "OPENVLA_BASE_PATH"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    candidates.extend(
        [
            path,
            Path("/sandbox/graph_internalization_exp/openvla_graph_internalization_rgb_runtime_complete_v3/checkpoints/openvla_7b_base"),
            Path("/sandbox/checkpoints/openvla_7b_base"),
        ]
    )
    for candidate in candidates:
        expanded = candidate.expanduser()
        if is_hf_checkpoint(expanded):
            return expanded
    raise RuntimeError("Could not locate a complete local Base OpenVLA checkpoint with config.json.")


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


class NpzCache:
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, path: str) -> Any:
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        arrays = np.load(path)
        self.cache[path] = arrays
        if len(self.cache) > self.capacity:
            _old_path, old_arrays = self.cache.popitem(last=False)
            try:
                old_arrays.close()
            except Exception:
                pass
        return arrays


def build_frame_rows(data_root: Path, split: str, frame_stride: int) -> list[dict[str, Any]]:
    rows = []
    for ep_row in read_jsonl(data_root / "manifests" / f"{split}.jsonl"):
        episode_dir = resolve_episode_dir(str(ep_row["episode_dir"]), data_root)
        meta = read_json(episode_dir / "metadata.json")
        frames_npz = episode_dir / "frames.npz"
        episode_length = int(meta["episode_length"])
        instruction = raw_instruction_from_metadata(meta)
        for frame_index in range(0, episode_length, frame_stride):
            rows.append(
                {
                    "split": split,
                    "episode_dir": str(episode_dir),
                    "frames_npz": str(frames_npz),
                    "frame_index": frame_index,
                    "task_id": int(meta["task_id"]),
                    "initial_state_id": int(meta.get("initial_state_id", -1)),
                    "policy_id": str(meta["policy_id"]),
                    "episode_id": str(meta["episode_id"]),
                    "episode_success": bool(meta.get("episode_success", False)),
                    "failure_category": str(meta.get("failure_category", "")),
                    "terminal_reason": str(meta.get("terminal_reason", "")),
                    "instruction": instruction,
                }
            )
    return rows


def build_balanced_epoch(train_rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    by_task: dict[int, dict[str, dict[bool, dict[str, list[dict[str, Any]]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for row in train_rows:
        by_task[int(row["task_id"])][str(row["policy_id"])][bool(row["episode_success"])][str(row["episode_id"])].append(row)
    tasks = sorted(by_task)
    sampled = []
    for _ in range(n):
        task = rng.choice(tasks)
        policy = rng.choice(sorted(by_task[task]))
        success = rng.choice(sorted(by_task[task][policy]))
        episode = rng.choice(sorted(by_task[task][policy][success]))
        sampled.append(rng.choice(by_task[task][policy][success][episode]))
    return sampled


def graph_from_targets(y_node: np.ndarray, y_edge: np.ndarray, ontology: dict[str, Any]) -> dict[str, Any]:
    idx_to_node = {meta["index"]: (node_id, meta) for node_id, meta in ontology["nodes"].items()}
    idx_to_pred = {idx: pred for pred, idx in ontology["predicates"].items()}
    nodes = []
    for idx, present in enumerate(y_node.astype(bool).tolist()):
        if present:
            node_id, meta = idx_to_node[idx]
            nodes.append({"id": node_id, "category": meta["category"], "entity_type": meta["entity_type"], "present": True})
    edges = []
    for i, j, r in np.argwhere(y_edge > 0.5):
        if i != j:
            edges.append({"subject": idx_to_node[int(i)][0], "predicate": idx_to_pred[int(r)], "object": idx_to_node[int(j)][0]})
    return {"nodes": sorted(nodes, key=lambda x: x["id"]), "binary_edges": sorted(edges, key=lambda x: (x["subject"], x["predicate"], x["object"]))}


def init_distributed():
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, world, local_rank, torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


def cleanup_distributed(world: int) -> None:
    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Frozen-OpenVLA Graph Generator training without feature cache.")
    parser.add_argument("--data-root", type=Path, default=Path("data/openvla_rollout_graph_v2"))
    parser.add_argument("--ontology", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/ontology/ontology.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rollout_depthfree_graph_generator_live_v1"))
    parser.add_argument("--architecture", default="rollout_live_depthfree_pooled_mlp_3d")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--xyz-weight", type=float, default=1.0)
    parser.add_argument("--train-samples-per-epoch", type=int, default=None)
    parser.add_argument("--limit-eval-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--feature-layer", type=int, default=-2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--xyz-sidecar-root", type=Path, default=None)
    parser.add_argument("--require-xyz-targets", action="store_true")
    parser.add_argument("--log-every-batches", type=int, default=10)
    args = parser.parse_args()

    rank, world, local_rank, device = init_distributed()
    report_dir = args.output_root / "reports"
    metric_dir = args.output_root / "metrics" / args.architecture
    ckpt_dir = args.output_root / "checkpoints" / args.architecture
    is_main = rank == 0
    report: dict[str, Any] = {"status": "started", "architecture": args.architecture, "live_openvla_forward": True}
    def log(message: str) -> None:
        if is_main:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

    try:
        import torch
        from PIL import Image
        from torch.nn.parallel import DistributedDataParallel as DDP

        from scene_graph_generator.graph_generator.losses_3d import graph_generator_3d_loss
        from scene_graph_generator.graph_generator.models.depth_augmented import OpenVLAOnlyPooledMLP3DGraphGenerator

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        rng = random.Random(args.seed)
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)

        args.checkpoint = resolve_checkpoint(args.checkpoint)
        log(f"loading frozen OpenVLA checkpoint: {args.checkpoint}")
        extractor = OpenVLAFeatureExtractor(args.checkpoint, device=str(device), dtype=args.dtype)
        ontology = read_json(args.ontology)
        validity_np = relation_validity_mask(ontology)
        validity = torch.tensor(validity_np, dtype=torch.bool, device=device)
        split_rows = {split: build_frame_rows(args.data_root, split, args.frame_stride) for split in ("train", "validation", "test")}
        eval_split_rows = {
            split: (rows[: args.limit_eval_frames] if args.limit_eval_frames is not None else rows)
            for split, rows in split_rows.items()
        }
        xyz_sidecar_root = args.xyz_sidecar_root or (args.data_root / "inspection" / "graph3d_positions_all")
        xyz_cache = RolloutXyzTargetCache(ontology, data_root=args.data_root, sidecar_root=xyz_sidecar_root)
        log(
            "split frames: "
            + ", ".join(f"{split}={len(rows)}" for split, rows in split_rows.items())
            + f"; xyz_sidecar_root={xyz_sidecar_root}"
        )
        k = len(ontology["nodes"])
        r = len(ontology["predicates"])
        input_dim = 4096
        model = OpenVLAOnlyPooledMLP3DGraphGenerator(input_dim, k, r, hidden_dim=1024, num_layers=3, dropout=0.1).to(device)
        ddp_model = DDP(model, device_ids=[local_rank]) if world > 1 else model
        opt = torch.optim.AdamW(ddp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        npz_cache = NpzCache()

        pos = torch.zeros(r, dtype=torch.float64)
        for row in split_rows["train"]:
            arrays = npz_cache.get(str(row["frames_npz"]))
            pos += torch.tensor(arrays["oracle_graph_tensor"][int(row["frame_index"])].sum(axis=(0, 1)), dtype=torch.float64)
        total = torch.tensor(float(len(split_rows["train"]) * k * k), dtype=torch.float64)
        pos_weight = ((total - pos).clamp_min(1.0) / pos.clamp_min(1.0)).clamp(1.0, 30.0).float().to(device)

        def build_batch(batch_rows: list[dict[str, Any]]):
            images = []
            instructions = []
            y_node = []
            y_edge = []
            y_xyz = []
            y_xyz_mask = []
            for row in batch_rows:
                arrays = npz_cache.get(str(row["frames_npz"]))
                idx = int(row["frame_index"])
                images.append(Image.fromarray(arrays["rgb"][idx]))
                instructions.append(str(row["instruction"]))
                y_node.append(torch.tensor(arrays["node_valid_mask"][idx], dtype=torch.float32, device=device))
                y_edge.append(torch.tensor(arrays["oracle_graph_tensor"][idx], dtype=torch.float32, device=device))
                xyz, xyz_mask, _xyz_source = xyz_cache.get(Path(row["episode_dir"]), arrays, idx)
                y_xyz.append(torch.tensor(xyz, dtype=torch.float32, device=device))
                y_xyz_mask.append(torch.tensor(xyz_mask, dtype=torch.float32, device=device))
            features, attn, token_type = extractor.extract_batch(images, instructions, feature_layer=args.feature_layer)
            return features, attn, token_type, torch.stack(y_node), torch.stack(y_edge), torch.stack(y_xyz), torch.stack(y_xyz_mask)

        def run_rows(split: str, rows_to_run: list[dict[str, Any]], train: bool, *, force_single_rank_full_eval: bool = False) -> dict[str, Any]:
            ddp_model.train(train)
            model.train(train)
            batch_size = args.batch_size if train else (args.eval_batch_size or args.batch_size)
            local_rows = rows_to_run if force_single_rank_full_eval else (rows_to_run[rank::world] if world > 1 else rows_to_run)
            active_model = model if force_single_rank_full_eval else ddp_model
            totals = {"loss": 0.0, "node_loss": 0.0, "edge_loss": 0.0, "xyz_loss": 0.0, "n": 0}
            examples = []
            xyz_pred_all = []
            xyz_gt_all = []
            xyz_mask_all = []
            schema_errors = 0
            num_batches = (len(local_rows) + batch_size - 1) // batch_size
            phase = "train" if train else "eval"
            for batch_idx, start in enumerate(range(0, len(local_rows), batch_size), start=1):
                batch = local_rows[start : start + batch_size]
                features, attn, token_type, y_node, y_edge, y_xyz, y_xyz_mask = build_batch(batch)
                with torch.set_grad_enabled(train):
                    out = active_model(features, attn, token_type)
                    losses = graph_generator_3d_loss(
                        out["node_logits"],
                        out["edge_logits"],
                        out["xyz"],
                        y_node,
                        y_edge,
                        y_xyz,
                        y_xyz_mask,
                        validity,
                        edge_pos_weight=pos_weight,
                        xyz_weight=args.xyz_weight,
                    )
                    if train:
                        opt.zero_grad(set_to_none=True)
                        losses["loss"].backward()
                        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                        opt.step()
                bsz = len(batch)
                for key in ("loss", "node_loss", "edge_loss", "xyz_loss"):
                    totals[key] += float(losses[key].detach().cpu()) * bsz
                totals["n"] += bsz
                if is_main and args.log_every_batches > 0 and (batch_idx == 1 or batch_idx % args.log_every_batches == 0 or batch_idx == num_batches):
                    avg_loss = totals["loss"] / max(totals["n"], 1)
                    avg_xyz = totals["xyz_loss"] / max(totals["n"], 1)
                    log(f"{phase}:{split} batch {batch_idx}/{num_batches} examples={totals['n']} loss={avg_loss:.6f} xyz_loss={avg_xyz:.6f}")
                if not train:
                    node_logits = out["node_logits"].detach().cpu().float().numpy()
                    edge_logits = out["edge_logits"].detach().cpu().float().numpy()
                    xyz_pred = out["xyz"].detach().cpu().float().numpy()
                    xyz_gt = y_xyz.detach().cpu().float().numpy()
                    xyz_mask = y_xyz_mask.detach().cpu().float().numpy()
                    xyz_pred_all.append(xyz_pred)
                    xyz_gt_all.append(xyz_gt)
                    xyz_mask_all.append(xyz_mask)
                    y_node_np = y_node.detach().cpu().float().numpy()
                    y_edge_np = y_edge.detach().cpu().float().numpy()
                    for i in range(len(batch)):
                        pred = decode_graph(node_logits[i], edge_logits[i], ontology, validity_np)
                        gt = graph_from_targets(y_node_np[i], y_edge_np[i], ontology)
                        schema_errors += len(validate_graph(pred))
                        examples.append(
                            {
                                "pred_nodes": graph_node_ids(pred),
                                "gt_nodes": graph_node_ids(gt),
                                "pred_edges": graph_triplets(pred),
                                "gt_edges": graph_triplets(gt),
                            }
                        )
            if world > 1 and not force_single_rank_full_eval:
                import torch.distributed as dist

                vec = torch.tensor([totals["loss"], totals["node_loss"], totals["edge_loss"], totals["xyz_loss"], totals["n"]], dtype=torch.float64, device=device)
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)
                totals = {"loss": float(vec[0]), "node_loss": float(vec[1]), "edge_loss": float(vec[2]), "xyz_loss": float(vec[3]), "n": int(vec[4])}
            losses_out = {key: totals[key] / max(totals["n"], 1) for key in ("loss", "node_loss", "edge_loss", "xyz_loss")}
            if train:
                return {**losses_out, "num_examples": totals["n"]}
            return {
                **losses_out,
                "num_examples": totals["n"],
                "schema_error_count": schema_errors,
                "metrics": summarize_examples(examples),
                "xyz_metrics": xyz_metrics(np.concatenate(xyz_pred_all), np.concatenate(xyz_gt_all), np.concatenate(xyz_mask_all)) if xyz_pred_all else {},
            }

        def selection_key(val_row: dict[str, Any]) -> tuple[float, float, float, float, float]:
            metrics = val_row["metrics"]
            xyz = val_row["xyz_metrics"]
            return (
                metrics["triplet"].get("macro_f1", 0.0),
                metrics["triplet"].get("f1", 0.0),
                metrics["graph"].get("exact_match", 0.0),
                -xyz.get("rmse", 999.0),
                -val_row["loss"],
            )

        history = []
        best_score = None
        best_state = None
        best_epoch = 0
        stale = 0
        started = time.time()
        samples_per_epoch = args.train_samples_per_epoch or len(split_rows["train"])
        if args.require_xyz_targets:
            train_xyz_points = 0
            for row in split_rows["train"]:
                arrays = npz_cache.get(str(row["frames_npz"]))
                _xyz, xyz_mask, _xyz_source = xyz_cache.get(Path(row["episode_dir"]), arrays, int(row["frame_index"]))
                train_xyz_points += int(np.asarray(xyz_mask).sum())
            if train_xyz_points <= 0:
                raise RuntimeError(
                    "No valid XYZ targets found for train split. "
                    f"Checked frames.npz and sidecars under {xyz_sidecar_root}."
                )
            log(f"train XYZ target check passed: valid_points={train_xyz_points}")
        for epoch in range(1, args.epochs + 1):
            log(f"epoch {epoch}/{args.epochs} started")
            epoch_rng = random.Random(args.seed + epoch)
            global_train_rows = build_balanced_epoch(split_rows["train"], samples_per_epoch, epoch_rng)
            train_row = run_rows("train", global_train_rows, train=True)
            val_row = run_rows("validation", eval_split_rows["validation"], train=False)
            score = selection_key(val_row)
            if is_main:
                history.append({"epoch": epoch, "train": train_row, "validation": val_row})
                write_json(metric_dir / "training_history.json", history)
                triplet = val_row["metrics"]["triplet"]
                graph = val_row["metrics"]["graph"]
                xyz = val_row["xyz_metrics"]
                log(
                    f"epoch {epoch} validation: loss={val_row['loss']:.6f} "
                    f"triplet_macro_f1={triplet.get('macro_f1', 0.0):.6f} "
                    f"triplet_f1={triplet.get('f1', 0.0):.6f} "
                    f"graph_em={graph.get('exact_match', 0.0):.6f} "
                    f"xyz_rmse={xyz.get('rmse', 999.0):.6f}"
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    stale = 0
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": best_state,
                            "epoch": epoch,
                            "ontology": ontology,
                            "history": history,
                            "openvla_dim": input_dim,
                            "depth_input_used": False,
                            "architecture": args.architecture,
                            "live_openvla_forward": True,
                            "openvla_checkpoint": str(args.checkpoint),
                        },
                        ckpt_dir / "best.pt",
                    )
                    log(f"new best checkpoint saved: epoch={epoch} path={ckpt_dir / 'best.pt'}")
                else:
                    stale += 1
                    log(f"no improvement: stale={stale}/{args.patience}")
            if world > 1:
                import torch.distributed as dist

                stale_tensor = torch.tensor([stale], dtype=torch.int64, device=device)
                dist.broadcast(stale_tensor, src=0)
                stale = int(stale_tensor.item())
            if stale >= args.patience:
                break

        if is_main:
            if best_state is None:
                raise RuntimeError("No checkpoint was selected.")
            model.load_state_dict(best_state)
        if world > 1:
            import torch.distributed as dist

            for param in model.parameters():
                dist.broadcast(param.data, src=0)
        if is_main:
            validation = run_rows("validation", eval_split_rows["validation"], train=False, force_single_rank_full_eval=True)
            test = run_rows("test", eval_split_rows["test"], train=False, force_single_rank_full_eval=True)
            summary = {
                "status": "ok",
                "architecture": args.architecture,
                "class_name": "OpenVLAOnlyPooledMLP3DGraphGenerator",
                "live_openvla_forward": True,
                "openvla_frozen": True,
                "depth_input_used": False,
                "openvla_checkpoint": str(args.checkpoint),
                "openvla_config_sha256": file_sha256(args.checkpoint / "config.json"),
                "world_size": world,
                "input_dim": input_dim,
                "hidden_dim": 1024,
                "num_layers": 3,
                "dropout": 0.1,
                "epochs_ran": len(history),
                "best_epoch": best_epoch,
                "validation": validation,
                "test": test,
                "checkpoint": str(ckpt_dir / "best.pt"),
                "checkpoint_sha256": file_sha256(ckpt_dir / "best.pt"),
                "xyz_sidecar_root": str(xyz_sidecar_root),
                "require_xyz_targets": bool(args.require_xyz_targets),
                "split_counts_frames": {split: len(split_rows[split]) for split in split_rows},
                "eval_counts_frames": {split: len(eval_split_rows[split]) for split in eval_split_rows},
                "limit_eval_frames": args.limit_eval_frames,
                "train_samples_per_epoch": samples_per_epoch,
                "balanced_sampler": "task -> policy_id -> episode_success -> episode_id -> frame",
                "validation_test_oversampling": False,
                "elapsed_sec": round(time.time() - started, 3),
                "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            }
            write_json(metric_dir / "validation_metrics.json", validation)
            write_json(metric_dir / "test_metrics.json", test)
            write_json(metric_dir / "training_summary.json", summary)
            report.update(summary)
        else:
            report["status"] = "ok"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        if is_main:
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / f"blocker_{args.architecture}.txt").write_text(report["traceback"], encoding="utf-8")
    finally:
        if is_main:
            write_json(report_dir / f"{args.architecture}_training_status.json", report)
            print(json.dumps({"status": report["status"], "summary": str(metric_dir / "training_summary.json")}, sort_keys=True))
        cleanup_distributed(world)
    if report["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
