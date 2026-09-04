#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
        if i == j:
            continue
        edges.append({"subject": idx_to_node[int(i)][0], "predicate": idx_to_pred[int(r)], "object": idx_to_node[int(j)][0]})
    return {"nodes": sorted(nodes, key=lambda x: x["id"]), "binary_edges": sorted(edges, key=lambda x: (x["subject"], x["predicate"], x["object"]))}


def graph_with_predicted_xyz(pred_graph: dict[str, Any], pred_xyz: np.ndarray, ontology: dict[str, Any]) -> dict[str, Any]:
    out = {
        "nodes": [dict(node) for node in pred_graph.get("nodes", [])],
        "binary_edges": [dict(edge) for edge in pred_graph.get("binary_edges", [])],
        "graph_type": "3d_scene_graph",
        "coordinate_frame": "mujoco_world",
    }
    for node in out["nodes"]:
        idx = ontology["nodes"][node["id"]]["index"]
        node["position_world_xyz"] = [float(x) for x in pred_xyz[idx]]
    return out


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
        by_policy = by_task[task]
        policy = rng.choice(sorted(by_policy))
        by_success = by_policy[policy]
        success = rng.choice(sorted(by_success))
        by_episode = by_success[success]
        episode = rng.choice(sorted(by_episode))
        sampled.append(rng.choice(by_episode[episode]))
    return sampled


def group_by_shard(rows: list[dict[str, Any]], rng: random.Random | None = None) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["shard"])].append(row)
    items = [(shard, group) for shard, group in groups.items()]
    if rng is not None:
        rng.shuffle(items)
        for _shard, group in items:
            rng.shuffle(group)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the locked depth-free 3-layer MLP Graph Generator on rollout data.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/openvla_rollout_graph_v2/openvla_feature_cache/base_openvla_rollout_v1"))
    parser.add_argument("--data-root", type=Path, default=Path("data/openvla_rollout_graph_v2"))
    parser.add_argument("--ontology", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/ontology/ontology.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/rollout_depthfree_graph_generator_v1"))
    parser.add_argument("--architecture", default="rollout_depthfree_pooled_mlp_3d")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--xyz-weight", type=float, default=1.0)
    parser.add_argument("--train-samples-per-epoch", type=int, default=None)
    parser.add_argument("--xyz-sidecar-root", type=Path, default=None)
    parser.add_argument("--require-xyz-targets", action="store_true")
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    report_dir = args.output_root / "reports"
    metric_dir = args.output_root / "metrics" / args.architecture
    ckpt_dir = args.output_root / "checkpoints" / args.architecture
    report: dict[str, Any] = {
        "status": "started",
        "architecture": args.architecture,
        "depth_input_used": False,
        "openvla_frozen": True,
    }
    def log(message: str) -> None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

    try:
        import torch
        from safetensors.torch import load_file

        from scene_graph_generator.graph_generator.losses_3d import graph_generator_3d_loss
        from scene_graph_generator.graph_generator.models.depth_augmented import OpenVLAOnlyPooledMLP3DGraphGenerator

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        rng = random.Random(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        ontology = read_json(args.ontology)
        validity_np = relation_validity_mask(ontology)
        validity = torch.tensor(validity_np, dtype=torch.bool, device="cuda")
        rows = read_jsonl(args.cache_dir / "cache_manifest.jsonl")
        split_rows = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
        if not split_rows["train"] or not split_rows["validation"] or not split_rows["test"]:
            raise RuntimeError({split: len(split_rows[split]) for split in split_rows})
        xyz_sidecar_root = args.xyz_sidecar_root or (args.data_root / "inspection" / "graph3d_positions_all")
        xyz_cache = RolloutXyzTargetCache(ontology, data_root=args.data_root, sidecar_root=xyz_sidecar_root)
        log(
            "cache rows: "
            + ", ".join(f"{split}={len(split_rows[split])}" for split in ("train", "validation", "test"))
            + f"; xyz_sidecar_root={xyz_sidecar_root}"
        )

        input_dim = int(rows[0]["feature_shape"][-1])
        k = len(ontology["nodes"])
        r = len(ontology["predicates"])
        model = OpenVLAOnlyPooledMLP3DGraphGenerator(input_dim, k, r, hidden_dim=1024, num_layers=3, dropout=0.1).to("cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        npz_cache = NpzCache()

        pos = torch.zeros(r, dtype=torch.float64)
        for row in split_rows["train"]:
            arrays = npz_cache.get(str(row["frames_npz"]))
            pos += torch.tensor(arrays["oracle_graph_tensor"][int(row["frame_index"])].sum(axis=(0, 1)), dtype=torch.float64)
        total = torch.tensor(float(len(split_rows["train"]) * k * k), dtype=torch.float64)
        pos_weight = ((total - pos).clamp_min(1.0) / pos.clamp_min(1.0)).clamp(1.0, 30.0).float().to("cuda")

        def pad_targets(batch: list[dict[str, Any]]):
            y_node = []
            y_edge = []
            y_xyz = []
            y_xyz_mask = []
            for row in batch:
                arrays = npz_cache.get(str(row["frames_npz"]))
                idx = int(row["frame_index"])
                y_node.append(torch.tensor(arrays["node_valid_mask"][idx], dtype=torch.float32))
                y_edge.append(torch.tensor(arrays["oracle_graph_tensor"][idx], dtype=torch.float32))
                xyz, xyz_mask, _xyz_source = xyz_cache.get(Path(row["episode_dir"]), arrays, idx)
                y_xyz.append(torch.tensor(xyz, dtype=torch.float32))
                y_xyz_mask.append(torch.tensor(xyz_mask, dtype=torch.float32))
            return torch.stack(y_node), torch.stack(y_edge), torch.stack(y_xyz), torch.stack(y_xyz_mask)

        def pad_features(batch: list[dict[str, Any]], tensors):
            max_len = max(tensors[f"{row['sample_key']}__features"].shape[0] for row in batch)
            b = len(batch)
            features = torch.zeros(b, max_len, input_dim, dtype=torch.bfloat16)
            attn = torch.zeros(b, max_len, dtype=torch.bool)
            token_type = torch.zeros(b, max_len, dtype=torch.long)
            for i, row in enumerate(batch):
                key = row["sample_key"]
                feat = tensors[f"{key}__features"]
                n = feat.shape[0]
                features[i, :n] = feat
                attn[i, :n] = tensors[f"{key}__attention_mask"].bool()
                token_type[i, :n] = tensors[f"{key}__token_type_mask"].long()
            return features.float(), attn, token_type

        def run_rows(split: str, rows_to_run: list[dict[str, Any]], train: bool) -> dict[str, Any]:
            model.train(train)
            totals = {"loss": 0.0, "node_loss": 0.0, "edge_loss": 0.0, "xyz_loss": 0.0, "n": 0}
            examples = []
            schema_errors = 0
            xyz_pred_all = []
            xyz_gt_all = []
            xyz_mask_all = []
            pred_root = args.output_root / "predictions" / args.architecture / split
            grouped = group_by_shard(rows_to_run, rng if train else None)
            total_batches = sum((len(shard_rows) + args.batch_size - 1) // args.batch_size for _shard_name, shard_rows in grouped)
            batch_index = 0
            phase = "train" if train else "eval"
            for shard_name, shard_rows in grouped:
                tensors = load_file(str(args.cache_dir / shard_name), device="cpu")
                for start in range(0, len(shard_rows), args.batch_size):
                    batch_index += 1
                    batch = shard_rows[start : start + args.batch_size]
                    features, attn, token_type = pad_features(batch, tensors)
                    y_node, y_edge, y_xyz, y_xyz_mask = pad_targets(batch)
                    features = features.to("cuda", non_blocking=True)
                    attn = attn.to("cuda", non_blocking=True)
                    token_type = token_type.to("cuda", non_blocking=True)
                    y_node = y_node.to("cuda", non_blocking=True)
                    y_edge = y_edge.to("cuda", non_blocking=True)
                    y_xyz = y_xyz.to("cuda", non_blocking=True)
                    y_xyz_mask = y_xyz_mask.to("cuda", non_blocking=True)
                    with torch.set_grad_enabled(train):
                        out = model(features, attn, token_type)
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
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            opt.step()
                    bsz = len(batch)
                    for key in ("loss", "node_loss", "edge_loss", "xyz_loss"):
                        totals[key] += float(losses[key].detach().cpu()) * bsz
                    totals["n"] += bsz
                    if args.log_every_batches > 0 and (batch_index == 1 or batch_index % args.log_every_batches == 0 or batch_index == total_batches):
                        avg_loss = totals["loss"] / max(totals["n"], 1)
                        avg_xyz = totals["xyz_loss"] / max(totals["n"], 1)
                        log(f"{phase}:{split} batch {batch_index}/{total_batches} examples={totals['n']} loss={avg_loss:.6f} xyz_loss={avg_xyz:.6f}")
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
                        for i, row in enumerate(batch):
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
                            if args.save_predictions:
                                pred3d = graph_with_predicted_xyz(pred, xyz_pred[i], ontology)
                                out_path = (
                                    pred_root
                                    / str(row["policy_id"])
                                    / f"task_{int(row['task_id']):02d}"
                                    / str(row["episode_id"])
                                    / f"{int(row['frame_index']):06d}.json"
                                )
                                write_json(out_path, pred3d)
            losses_out = {key: totals[key] / max(totals["n"], 1) for key in ("loss", "node_loss", "edge_loss", "xyz_loss")}
            if train:
                return {**losses_out, "num_examples": totals["n"]}
            xyz_summary = xyz_metrics(np.concatenate(xyz_pred_all), np.concatenate(xyz_gt_all), np.concatenate(xyz_mask_all)) if xyz_pred_all else {}
            return {
                **losses_out,
                "num_examples": totals["n"],
                "schema_error_count": schema_errors,
                "metrics": summarize_examples(examples),
                "xyz_metrics": xyz_summary,
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
            train_epoch_rows = build_balanced_epoch(split_rows["train"], samples_per_epoch, rng)
            train_row = run_rows("train", train_epoch_rows, train=True)
            val_row = run_rows("validation", split_rows["validation"], train=False)
            score = selection_key(val_row)
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
                        "config": {
                            "class_name": "OpenVLAOnlyPooledMLP3DGraphGenerator",
                            "hidden_dim": 1024,
                            "num_layers": 3,
                            "dropout": 0.1,
                            "learning_rate": args.learning_rate,
                            "weight_decay": args.weight_decay,
                            "batch_size": args.batch_size,
                            "optimizer": "AdamW",
                            "gradient_clip_norm": 1.0,
                            "early_stopping_patience": args.patience,
                            "xyz_weight": args.xyz_weight,
                            "balanced_sampler": "task -> policy_id -> episode_success -> episode_id -> frame",
                        },
                    },
                    ckpt_dir / "best.pt",
                )
                log(f"new best checkpoint saved: epoch={epoch} path={ckpt_dir / 'best.pt'}")
            else:
                stale += 1
                log(f"no improvement: stale={stale}/{args.patience}")
                if stale >= args.patience:
                    break

        if best_state is None:
            raise RuntimeError("No checkpoint was selected.")
        model.load_state_dict(best_state)
        validation = run_rows("validation", split_rows["validation"], train=False)
        test = run_rows("test", split_rows["test"], train=False)
        summary = {
            "status": "ok",
            "architecture": args.architecture,
            "class_name": "OpenVLAOnlyPooledMLP3DGraphGenerator",
            "depth_input_used": False,
            "openvla_frozen": True,
            "openvla_forward_used_during_training": False,
            "input_dim": input_dim,
            "hidden_dim": 1024,
            "num_layers": 3,
            "dropout": 0.1,
            "epochs_ran": len(history),
            "best_epoch": best_epoch,
            "best_validation_selection_key": list(best_score) if best_score is not None else None,
            "validation": validation,
            "test": test,
            "checkpoint": str(ckpt_dir / "best.pt"),
            "checkpoint_sha256": file_sha256(ckpt_dir / "best.pt"),
            "ontology": str(args.ontology),
            "cache_dir": str(args.cache_dir),
            "xyz_sidecar_root": str(xyz_sidecar_root),
            "require_xyz_targets": bool(args.require_xyz_targets),
            "split_counts_frames": {split: len(split_rows[split]) for split in split_rows},
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
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"blocker_{args.architecture}.txt").write_text(report["traceback"], encoding="utf-8")
    write_json(report_dir / f"{args.architecture}_training_status.json", report)
    print(json.dumps({"status": report["status"], "summary": str(metric_dir / "training_summary.json")}, sort_keys=True))
    if report["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
