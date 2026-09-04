#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from .cached_eval_common import collect_probability_records, score_records, write_markdown_summary
    from .common import DEFAULT_OUTPUT_ROOT
except ImportError:  # pragma: no cover - CLI path execution
    from cached_eval_common import collect_probability_records, score_records, write_markdown_summary
    from common import DEFAULT_OUTPUT_ROOT
from scene_graph_generator.graph_generator.schema import read_json, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "feature_cache" / "all_frames")
    ap.add_argument("--architecture", choices=["pooled_mlp", "node_query_decoder"], default="pooled_mlp")
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--save-predictions", action="store_true")
    args = ap.parse_args()

    started = time.time()
    threshold_path = args.output_root / "metrics" / args.architecture / "selected_thresholds.json"
    if not threshold_path.exists():
        raise FileNotFoundError(f"Missing selected thresholds: {threshold_path}")
    thresholds = read_json(threshold_path)
    bundle = collect_probability_records(
        architecture=args.architecture,
        cache_dir=args.cache_dir,
        output_root=args.output_root,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        max_examples=args.max_examples,
    )
    pred_root = args.output_root / "predictions" / args.architecture / args.split if args.save_predictions else None
    summary = score_records(
        bundle["records"],
        bundle["ontology"],
        bundle["validity_mask"],
        node_threshold=float(thresholds["node_threshold"]),
        predicate_thresholds=thresholds["predicate_thresholds"],
        include_confidence=True,
        save_predictions_root=pred_root,
    )
    report = {
        "status": "ok",
        "architecture": args.architecture,
        "split": args.split,
        "num_examples": len(bundle["records"]),
        "checkpoint": bundle["checkpoint_path"],
        "checkpoint_epoch": bundle["checkpoint_epoch"],
        "thresholds": str(threshold_path),
        "predictions": str(pred_root) if pred_root else None,
        "openvla_forward_used": False,
        "elapsed_sec": round(time.time() - started, 3),
        **summary,
    }
    out_path = args.output_root / "metrics" / args.architecture / f"{args.split}_metrics.json"
    write_json(out_path, report)
    write_markdown_summary(args.output_root / "reports" / f"{args.architecture}_{args.split}_evaluation.md", f"{args.architecture} {args.split} Evaluation", report)
    print(json.dumps({"status": "ok", "architecture": args.architecture, "split": args.split, "output": str(out_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
