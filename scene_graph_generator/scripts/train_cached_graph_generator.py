#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph_generator.graph_generator.decoding import decode_graph
from scene_graph_generator.graph_generator.masks import relation_validity_mask
from scene_graph_generator.graph_generator.metrics import summarize_examples
from scene_graph_generator.graph_generator.schema import compact_graph, graph_node_ids, graph_triplets, read_json, validate_graph, write_json
from scene_graph_generator.graph_generator.targets import encode_targets


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pad_batch(items, tensors):
    import torch

    max_len = max(tensors[f"{x['sample_key']}__features"].shape[0] for x in items)
    dim = tensors[f"{items[0]['sample_key']}__features"].shape[1]
    b = len(items)
    features = torch.zeros(b, max_len, dim, dtype=torch.bfloat16)
    attn = torch.zeros(b, max_len, dtype=torch.bool)
    token_type = torch.zeros(b, max_len, dtype=torch.long)
    y_node = torch.stack([x["y_node"] for x in items])
    y_edge = torch.stack([x["y_edge"] for x in items])
    for i, item in enumerate(items):
        key = item["sample_key"]
        feat = tensors[f"{key}__features"]
        n = feat.shape[0]
        features[i, :n] = feat
        attn[i, :n] = tensors[f"{key}__attention_mask"].bool()
        token_type[i, :n] = tensors[f"{key}__token_type_mask"].long()
    return features.float(), attn, token_type, y_node, y_edge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", choices=["pooled_mlp", "node_query_decoder"], required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/feature_cache/all_frames"))
    ap.add_argument("--output-root", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()
    report = {"status": "started", "architecture": args.architecture, "epochs": args.epochs, "batch_size": args.batch_size}
    report_dir = args.output_root / "reports"
    metric_dir = args.output_root / "metrics" / args.architecture
    ckpt_dir = args.output_root / "checkpoints" / args.architecture
    try:
        import torch
        from safetensors.torch import load_file

        from scene_graph_generator.graph_generator.losses import graph_generator_loss
        from scene_graph_generator.graph_generator.models.node_query_decoder import NodeQueryGraphDecoder
        from scene_graph_generator.graph_generator.models.pooled_mlp import PooledMLPGraphGenerator

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        ontology = read_json(args.output_root / "ontology" / "ontology.json")
        validity_np = relation_validity_mask(ontology)
        validity = torch.tensor(validity_np, dtype=torch.bool, device="cuda")
        rows = list(read_jsonl(args.cache_dir / "cache_manifest.jsonl"))
        row_by_key = {}
        split_by_shard = defaultdict(lambda: defaultdict(list))
        y_edges_train = []
        for row in rows:
            graph = compact_graph(read_json(Path(row["graph_path"])))
            y_node_np, y_edge_np = encode_targets(graph, ontology)
            item = {
                **row,
                "gt_graph": graph,
                "y_node": torch.tensor(y_node_np, dtype=torch.float32),
                "y_edge": torch.tensor(y_edge_np, dtype=torch.float32),
            }
            row_by_key[row["sample_key"]] = item
            split_by_shard[row["split"]][row["shard"]].append(item)
            if row["split"] == "train":
                y_edges_train.append(item["y_edge"])
        y_edges = torch.stack(y_edges_train)
        pos = y_edges.sum(dim=(0, 1, 2)).clamp_min(1.0)
        total = torch.tensor(float(y_edges.shape[0] * y_edges.shape[1] * y_edges.shape[2]))
        neg = (total - pos).clamp_min(1.0)
        pos_weight = (neg / pos).clamp(1.0, 30.0).to("cuda")
        input_dim = int(next(iter(read_jsonl(args.cache_dir / "cache_manifest.jsonl")))["feature_shape"][-1])
        k = len(ontology["nodes"])
        r = len(ontology["predicates"])
        if args.architecture == "pooled_mlp":
            model = PooledMLPGraphGenerator(input_dim, k, r, hidden_dim=1024, num_layers=3, dropout=0.1)
        else:
            model = NodeQueryGraphDecoder(input_dim, k, r, model_dim=512, num_decoder_layers=4, num_attention_heads=8, feedforward_dim=2048, dropout=0.1)
        model = model.to("cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

        def run_split(split, train=False, save_predictions=False):
            model.train(train)
            totals = {"loss": 0.0, "node_loss": 0.0, "edge_loss": 0.0, "n": 0}
            examples = []
            schema_errors = 0
            pred_root = args.output_root / "predictions" / args.architecture / split
            shard_names = sorted(split_by_shard[split])
            if train:
                random.shuffle(shard_names)
            for shard_name in shard_names:
                shard_items = list(split_by_shard[split][shard_name])
                if train:
                    random.shuffle(shard_items)
                tensors = load_file(str(args.cache_dir / shard_name), device="cpu")
                for start in range(0, len(shard_items), args.batch_size):
                    batch = shard_items[start : start + args.batch_size]
                    features, attn, token_type, y_node, y_edge = pad_batch(batch, tensors)
                    features = features.to("cuda", non_blocking=True)
                    attn = attn.to("cuda", non_blocking=True)
                    token_type = token_type.to("cuda", non_blocking=True)
                    y_node = y_node.to("cuda", non_blocking=True)
                    y_edge = y_edge.to("cuda", non_blocking=True)
                    with torch.set_grad_enabled(train):
                        if args.architecture == "pooled_mlp":
                            out = model(features, attn, token_type)
                        else:
                            out = model(features, attn)
                        losses = graph_generator_loss(out["node_logits"], out["edge_logits"], y_node, y_edge, validity, edge_pos_weight=pos_weight)
                        if train:
                            opt.zero_grad(set_to_none=True)
                            losses["loss"].backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            opt.step()
                    bsz = len(batch)
                    totals["loss"] += float(losses["loss"].detach().cpu()) * bsz
                    totals["node_loss"] += float(losses["node_loss"].cpu()) * bsz
                    totals["edge_loss"] += float(losses["edge_loss"].cpu()) * bsz
                    totals["n"] += bsz
                    if not train:
                        node_logits = out["node_logits"].detach().cpu().float().numpy()
                        edge_logits = out["edge_logits"].detach().cpu().float().numpy()
                        for i, item in enumerate(batch):
                            pred = decode_graph(node_logits[i], edge_logits[i], ontology, validity_np)
                            schema_errors += len(validate_graph(pred))
                            gt = item["gt_graph"]
                            examples.append(
                                {
                                    "pred_nodes": graph_node_ids(pred),
                                    "gt_nodes": graph_node_ids(gt),
                                    "pred_edges": graph_triplets(pred),
                                    "gt_edges": graph_triplets(gt),
                                }
                            )
                            if save_predictions:
                                out_path = pred_root / f"task_{item['task_id']:02d}" / f"global_{item['global_episode_index']:06d}" / f"{item['frame_index']:06d}.json"
                                write_json(out_path, pred)
            losses_out = {k: totals[k] / max(totals["n"], 1) for k in ("loss", "node_loss", "edge_loss")}
            if train:
                return {**losses_out, "num_examples": totals["n"]}
            return {**losses_out, "schema_error_count": schema_errors, "metrics": summarize_examples(examples)}

        history = []
        def selection_key(val_row):
            metrics = val_row["metrics"]
            return (
                metrics["triplet"].get("macro_f1", 0.0),
                metrics["triplet"].get("f1", 0.0),
                metrics["graph"].get("exact_match", 0.0),
                metrics["node"].get("f1", 0.0),
                -val_row["loss"],
            )

        best_score = None
        best_state = None
        best_epoch = 0
        stale = 0
        started = time.time()
        for epoch in range(1, args.epochs + 1):
            train_row = run_split("train", train=True)
            val_row = run_split("validation", train=False)
            score = selection_key(val_row)
            epoch_row = {"epoch": epoch, "train": train_row, "validation": val_row}
            history.append(epoch_row)
            write_json(metric_dir / "training_history.json", history)
            if best_score is None or score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                stale = 0
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": best_state, "epoch": epoch, "ontology": ontology, "history": history}, ckpt_dir / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best_state)
        val_final = run_split("validation", train=False, save_predictions=True)
        summary = {
            "status": "ok",
            "architecture": args.architecture,
            "epochs_ran": len(history),
            "best_epoch": best_epoch,
            "best_validation_selection_key": list(best_score) if best_score is not None else None,
            "best_validation_triplet_macro_f1": best_score[0] if best_score is not None else None,
            "validation": val_final,
            "checkpoint": str(ckpt_dir / "best.pt"),
            "elapsed_sec": round(time.time() - started, 3),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "openvla_forward_used": False,
        }
        write_json(metric_dir / "validation_metrics.json", val_final)
        write_json(metric_dir / "training_summary.json", summary)
        report.update(summary)
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        blocker = ["# Blocker: Full Training", "", f"- Architecture: `{args.architecture}`", "", "```", report["traceback"], "```"]
        (report_dir / f"blocker_full_training_{args.architecture}.md").write_text("\n".join(blocker))
    write_json(report_dir / f"{args.architecture}_training_status.json", report)
    print(json.dumps({"status": report["status"], "architecture": args.architecture, "summary": str(metric_dir / "training_summary.json")}, sort_keys=True))
    if report["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
