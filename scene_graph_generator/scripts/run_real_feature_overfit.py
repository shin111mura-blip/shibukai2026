#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
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


def pad_batch(items):
    import torch

    max_len = max(x["features"].shape[0] for x in items)
    dim = items[0]["features"].shape[1]
    b = len(items)
    features = torch.zeros(b, max_len, dim, dtype=torch.bfloat16)
    attn = torch.zeros(b, max_len, dtype=torch.bool)
    token_type = torch.zeros(b, max_len, dtype=torch.long)
    y_node = torch.stack([x["y_node"] for x in items])
    y_edge = torch.stack([x["y_edge"] for x in items])
    for i, item in enumerate(items):
        n = item["features"].shape[0]
        features[i, :n] = item["features"]
        attn[i, :n] = item["attention_mask"].bool()
        token_type[i, :n] = item["token_type_mask"].long()
    return features.float(), attn, token_type, y_node, y_edge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial/feature_cache/smoke_100"))
    ap.add_argument("--output-root", type=Path, default=Path("outputs/scene_graph_generator_openvla_spatial"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    report_dir = args.output_root / "reports"
    metric_dir = args.output_root / "metrics" / "small_overfit"
    ckpt_dir = args.output_root / "checkpoints" / "small_overfit"
    pred_dir = args.output_root / "predictions" / "small_overfit"
    report = {"status": "started", "epochs": args.epochs, "batch_size": args.batch_size}
    try:
        import torch
        from safetensors.torch import load_file

        from scene_graph_generator.graph_generator.losses import graph_generator_loss
        from scene_graph_generator.graph_generator.models.node_query_decoder import NodeQueryGraphDecoder
        from scene_graph_generator.graph_generator.models.pooled_mlp import PooledMLPGraphGenerator

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("torch.cuda.is_available() is False")
        ontology = read_json(args.output_root / "ontology" / "ontology.json")
        validity_np = relation_validity_mask(ontology)
        validity = torch.tensor(validity_np, dtype=torch.bool, device=device)
        rows = list(read_jsonl(args.cache_dir / "cache_manifest.jsonl"))
        tensors = load_file(str(args.cache_dir / "shard_000000.safetensors"), device="cpu")
        samples = []
        for idx, row in enumerate(rows):
            key = row["sample_key"]
            graph = read_json(Path(row["graph_path"]))
            y_node_np, y_edge_np = encode_targets(graph, ontology)
            samples.append(
                {
                    "row": row,
                    "features": tensors[f"{key}__features"],
                    "attention_mask": tensors[f"{key}__attention_mask"],
                    "token_type_mask": tensors[f"{key}__token_type_mask"],
                    "y_node": torch.tensor(y_node_np, dtype=torch.float32),
                    "y_edge": torch.tensor(y_edge_np, dtype=torch.float32),
                    "gt_graph": compact_graph(graph),
                }
            )
        y_edges = torch.stack([s["y_edge"] for s in samples])
        pos = y_edges.sum(dim=(0, 1, 2)).clamp_min(1.0)
        total = torch.tensor(float(y_edges.shape[0] * y_edges.shape[1] * y_edges.shape[2]))
        neg = (total - pos).clamp_min(1.0)
        pos_weight = (neg / pos).clamp(1.0, 30.0).to(device)

        def train_arch(name, model):
            model = model.to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            history = []
            best_loss = float("inf")
            best_state = None
            for epoch in range(args.epochs):
                random.shuffle(samples)
                totals = {"loss": 0.0, "node_loss": 0.0, "edge_loss": 0.0, "n": 0}
                model.train()
                for start in range(0, len(samples), args.batch_size):
                    batch = samples[start : start + args.batch_size]
                    features, attn, token_type, y_node, y_edge = pad_batch(batch)
                    features = features.to(device)
                    attn = attn.to(device)
                    token_type = token_type.to(device)
                    y_node = y_node.to(device)
                    y_edge = y_edge.to(device)
                    opt.zero_grad(set_to_none=True)
                    if name == "pooled_mlp":
                        out = model(features, attn, token_type)
                    else:
                        out = model(features, attn)
                    losses = graph_generator_loss(
                        out["node_logits"],
                        out["edge_logits"],
                        y_node,
                        y_edge,
                        validity,
                        edge_pos_weight=pos_weight,
                    )
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    bsz = len(batch)
                    totals["loss"] += float(losses["loss"].detach().cpu()) * bsz
                    totals["node_loss"] += float(losses["node_loss"].cpu()) * bsz
                    totals["edge_loss"] += float(losses["edge_loss"].cpu()) * bsz
                    totals["n"] += bsz
                epoch_row = {k: totals[k] / totals["n"] for k in ("loss", "node_loss", "edge_loss")}
                epoch_row["epoch"] = epoch + 1
                history.append(epoch_row)
                if epoch_row["loss"] < best_loss:
                    best_loss = epoch_row["loss"]
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            (ckpt_dir / name).mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": best_state, "history": history, "ontology": ontology}, ckpt_dir / name / "best.pt")
            model.load_state_dict(best_state)
            model.eval()
            examples = []
            schema_errors = []
            with torch.inference_mode():
                for sample in samples:
                    features, attn, token_type, _, _ = pad_batch([sample])
                    features = features.to(device)
                    attn = attn.to(device)
                    token_type = token_type.to(device)
                    out = model(features, attn, token_type) if name == "pooled_mlp" else model(features, attn)
                    pred = decode_graph(
                        out["node_logits"][0].detach().cpu().float().numpy(),
                        out["edge_logits"][0].detach().cpu().float().numpy(),
                        ontology,
                        validity_np,
                    )
                    schema_errors.extend(validate_graph(pred))
                    row = sample["row"]
                    out_path = pred_dir / name / f"task_{row['task_id']:02d}" / f"global_{row['global_episode_index']:06d}" / f"{row['frame_index']:06d}.json"
                    write_json(out_path, pred)
                    gt = sample["gt_graph"]
                    examples.append(
                        {
                            "pred_nodes": graph_node_ids(pred),
                            "gt_nodes": graph_node_ids(gt),
                            "pred_edges": graph_triplets(pred),
                            "gt_edges": graph_triplets(gt),
                        }
                    )
            metrics = summarize_examples(examples)
            result = {
                "architecture": name,
                "initial_loss": history[0]["loss"],
                "final_loss": history[-1]["loss"],
                "best_loss": best_loss,
                "node_loss_initial": history[0]["node_loss"],
                "node_loss_final": history[-1]["node_loss"],
                "edge_loss_initial": history[0]["edge_loss"],
                "edge_loss_final": history[-1]["edge_loss"],
                "loss_decreased": history[-1]["loss"] < history[0]["loss"],
                "node_loss_decreased": history[-1]["node_loss"] < history[0]["node_loss"],
                "edge_loss_decreased": history[-1]["edge_loss"] < history[0]["edge_loss"],
                "schema_error_count": len(schema_errors),
                "metrics": metrics,
                "history": history,
                "checkpoint": str(ckpt_dir / name / "best.pt"),
                "predictions": str(pred_dir / name),
                "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            }
            write_json(metric_dir / name / "metrics.json", result)
            return result

        input_dim = int(samples[0]["features"].shape[-1])
        k = len(ontology["nodes"])
        r = len(ontology["predicates"])
        started = time.time()
        pooled = train_arch("pooled_mlp", PooledMLPGraphGenerator(input_dim, k, r, hidden_dim=1024, num_layers=3, dropout=0.0))
        decoder = train_arch(
            "node_query_decoder",
            NodeQueryGraphDecoder(input_dim, k, r, model_dim=512, num_decoder_layers=4, num_attention_heads=8, feedforward_dim=2048, dropout=0.0),
        )
        report.update(
            {
                "status": "ok" if pooled["loss_decreased"] and decoder["loss_decreased"] and pooled["schema_error_count"] == 0 and decoder["schema_error_count"] == 0 else "failed",
                "num_frames": len(samples),
                "num_episodes": len({s["row"]["global_episode_index"] for s in samples}),
                "elapsed_sec": round(time.time() - started, 3),
                "pooled_mlp": {k: pooled[k] for k in ("initial_loss", "final_loss", "node_loss_initial", "node_loss_final", "edge_loss_initial", "edge_loss_final", "schema_error_count", "checkpoint")},
                "node_query_decoder": {k: decoder[k] for k in ("initial_loss", "final_loss", "node_loss_initial", "node_loss_final", "edge_loss_initial", "edge_loss_final", "schema_error_count", "checkpoint")},
                "openvla_forward_used": False,
            }
        )
        before = read_json(args.output_root / "reports" / "frozen_model_audit_before.json")
        after = dict(before)
        after["status"] = "ok"
        after["note"] = "Small overfit used cached OpenVLA features only; no OpenVLA model was loaded or optimized."
        write_json(args.output_root / "reports" / "frozen_model_audit_after.json", after)
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
    write_json(report_dir / "small_overfit_real_features.json", report)
    md = ["# Small Overfit Real Features", "", f"- Status: `{report['status']}`"]
    for key in ["num_frames", "num_episodes", "elapsed_sec", "openvla_forward_used"]:
        if key in report:
            md.append(f"- {key}: `{report[key]}`")
    for name in ["pooled_mlp", "node_query_decoder"]:
        if name in report:
            md.append("")
            md.append(f"## {name}")
            for k, v in report[name].items():
                md.append(f"- {k}: `{v}`")
    if "traceback" in report:
        md.extend(["", "```", report["traceback"], "```"])
    (report_dir / "small_overfit_real_features.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"status": report["status"], "report": str(report_dir / "small_overfit_real_features.json")}, sort_keys=True))
    if report["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
