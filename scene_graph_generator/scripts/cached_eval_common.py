from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph_generator.graph_generator.decoding import decode_graph
from scene_graph_generator.graph_generator.masks import relation_validity_mask
from scene_graph_generator.graph_generator.metrics import summarize_examples
from scene_graph_generator.graph_generator.schema import compact_graph, graph_node_ids, graph_triplets, read_json, validate_graph, write_json
from scene_graph_generator.graph_generator.targets import encode_targets


def read_jsonl_iter(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_cache_rows(cache_dir: Path, output_root: Path, split: str) -> List[Dict[str, Any]]:
    ontology = read_json(output_root / "ontology" / "ontology.json")
    rows = []
    for row in read_jsonl_iter(cache_dir / "cache_manifest.jsonl"):
        if row["split"] != split:
            continue
        graph = compact_graph(read_json(Path(row["graph_path"])))
        y_node, y_edge = encode_targets(graph, ontology)
        rows.append({**row, "gt_graph": graph, "y_node_np": y_node, "y_edge_np": y_edge})
    return rows


def rows_by_shard(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    by_shard: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shard[str(row["shard"])].append(row)
    return dict(by_shard)


def build_model(architecture: str, input_dim: int, ontology: Mapping[str, Any]):
    from scene_graph_generator.graph_generator.models.node_query_decoder import NodeQueryGraphDecoder
    from scene_graph_generator.graph_generator.models.pooled_mlp import PooledMLPGraphGenerator

    k = len(ontology["nodes"])
    r = len(ontology["predicates"])
    if architecture == "pooled_mlp":
        return PooledMLPGraphGenerator(input_dim, k, r, hidden_dim=1024, num_layers=3, dropout=0.1)
    if architecture == "node_query_decoder":
        return NodeQueryGraphDecoder(input_dim, k, r, model_dim=512, num_decoder_layers=4, num_attention_heads=8, feedforward_dim=2048, dropout=0.1)
    raise ValueError(f"Unsupported architecture: {architecture}")


def load_trained_model(architecture: str, output_root: Path, input_dim: int, ontology: Mapping[str, Any], device: str):
    import torch

    checkpoint_path = output_root / "checkpoints" / architecture / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    model = build_model(architecture, input_dim, ontology).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, checkpoint_path


def pad_batch(items, tensors):
    import torch

    max_len = max(tensors[f"{x['sample_key']}__features"].shape[0] for x in items)
    dim = tensors[f"{items[0]['sample_key']}__features"].shape[1]
    features = torch.zeros(len(items), max_len, dim, dtype=torch.bfloat16)
    attn = torch.zeros(len(items), max_len, dtype=torch.bool)
    token_type = torch.zeros(len(items), max_len, dtype=torch.long)
    for i, item in enumerate(items):
        key = item["sample_key"]
        feat = tensors[f"{key}__features"]
        n = feat.shape[0]
        features[i, :n] = feat
        attn[i, :n] = tensors[f"{key}__attention_mask"].bool()
        token_type[i, :n] = tensors[f"{key}__token_type_mask"].long()
    return features.float(), attn, token_type


def iter_model_outputs(
    *,
    architecture: str,
    model,
    rows: List[Mapping[str, Any]],
    cache_dir: Path,
    batch_size: int,
    device: str,
):
    import torch
    from safetensors.torch import load_file

    for shard_name, shard_items in sorted(rows_by_shard(rows).items()):
        tensors = load_file(str(cache_dir / shard_name), device="cpu")
        for start in range(0, len(shard_items), batch_size):
            batch = list(shard_items[start : start + batch_size])
            features, attn, token_type = pad_batch(batch, tensors)
            features = features.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            token_type = token_type.to(device, non_blocking=True)
            with torch.inference_mode():
                if architecture == "pooled_mlp":
                    out = model(features, attn, token_type)
                else:
                    out = model(features, attn)
            yield batch, out["node_logits"].detach().cpu().float().numpy(), out["edge_logits"].detach().cpu().float().numpy()


def collect_probability_records(
    *,
    architecture: str,
    cache_dir: Path,
    output_root: Path,
    split: str,
    batch_size: int,
    device: str,
    max_examples: int | None = None,
) -> Dict[str, Any]:
    import torch

    ontology = read_json(output_root / "ontology" / "ontology.json")
    rows = load_cache_rows(cache_dir, output_root, split)
    if max_examples is not None:
        rows = rows[:max_examples]
    if not rows:
        raise RuntimeError(f"No rows found for split={split}")
    input_dim = int(rows[0]["feature_shape"][-1])
    model, ckpt, checkpoint_path = load_trained_model(architecture, output_root, input_dim, ontology, device)
    records = []
    for batch, node_logits, edge_logits in iter_model_outputs(
        architecture=architecture,
        model=model,
        rows=rows,
        cache_dir=cache_dir,
        batch_size=batch_size,
        device=device,
    ):
        node_probs = 1.0 / (1.0 + np.exp(-node_logits))
        edge_probs = 1.0 / (1.0 + np.exp(-edge_logits))
        for i, row in enumerate(batch):
            records.append(
                {
                    "row": row,
                    "node_probs": node_probs[i],
                    "edge_probs": edge_probs[i],
                }
            )
    return {
        "ontology": ontology,
        "validity_mask": relation_validity_mask(ontology),
        "records": records,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }


def logits_from_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return np.log(probs / (1.0 - probs))


def score_records(
    records: Iterable[Mapping[str, Any]],
    ontology: Mapping[str, Any],
    validity_mask: np.ndarray,
    *,
    node_threshold: float,
    predicate_thresholds: Mapping[str, float],
    include_confidence: bool = False,
    save_predictions_root: Path | None = None,
) -> Dict[str, Any]:
    examples = []
    schema_errors = 0
    for rec in records:
        row = rec["row"]
        pred = decode_graph(
            logits_from_probs(rec["node_probs"]),
            logits_from_probs(rec["edge_probs"]),
            dict(ontology),
            validity_mask,
            node_threshold=node_threshold,
            predicate_thresholds=predicate_thresholds,
            include_confidence=include_confidence,
        )
        schema_errors += len(validate_graph(pred))
        gt = row["gt_graph"]
        examples.append(
            {
                "pred_nodes": graph_node_ids(pred),
                "gt_nodes": graph_node_ids(gt),
                "pred_edges": graph_triplets(pred),
                "gt_edges": graph_triplets(gt),
            }
        )
        if save_predictions_root is not None:
            payload = {
                **pred,
                "metadata": {
                    "task_id": row["task_id"],
                    "global_episode_index": row["global_episode_index"],
                    "frame_index": row["frame_index"],
                    "split": row["split"],
                    "sample_key": row["sample_key"],
                    "image_path": row.get("image_path"),
                    "graph_path": row["graph_path"],
                },
            }
            out_path = (
                save_predictions_root
                / f"task_{row['task_id']:02d}"
                / f"global_{row['global_episode_index']:06d}"
                / f"{row['frame_index']:06d}.json"
            )
            write_json(out_path, payload)
    return {"schema_error_count": schema_errors, "metrics": summarize_examples(examples)}


def write_markdown_summary(path: Path, title: str, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = summary.get("metrics", summary)
    lines = [f"# {title}", ""]
    for family in ("node", "triplet", "graph"):
        if family in metrics:
            lines.append(f"## {family}")
            for key, value in metrics[family].items():
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:.6f}")
                else:
                    lines.append(f"- {key}: {value}")
            lines.append("")
    path.write_text("\n".join(lines))


def sample_records(records: List[Mapping[str, Any]], count: int, seed: int) -> List[Mapping[str, Any]]:
    rng = random.Random(seed)
    if len(records) <= count:
        return list(records)
    return rng.sample(records, count)
