#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from .common import DEFAULT_OUTPUT_ROOT
except ImportError:  # pragma: no cover - CLI path execution
    from common import DEFAULT_OUTPUT_ROOT
from scene_graph_generator.graph_generator.feature_cache import estimate_feature_bytes, read_jsonl
from scene_graph_generator.graph_generator.schema import write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_ROOT / "manifests" / "all_frames.jsonl")
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--dry-run-estimate", action="store_true")
    args = ap.parse_args()
    rows = read_jsonl(args.manifest)
    if args.dry_run_estimate:
        estimate = {
            "num_frames": len(rows),
            "assumed_tokens_per_frame": 128,
            "assumed_hidden_dim": 4096,
            "dtype": "bfloat16",
            "estimated_bytes": estimate_feature_bytes(len(rows), 128, 4096, 2),
            "status": "estimate_only",
        }
        write_json(args.output_root / "reports" / "feature_cache_estimate.json", estimate)
        print(json.dumps(estimate, sort_keys=True))
        return
    try:
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception as exc:
        report = {"status": "blocked", "reason": f"Missing runtime dependency: {exc}", "requested_limit": args.limit}
        write_json(args.output_root / "reports" / "feature_cache_smoke.json", report)
        print(json.dumps(report, sort_keys=True))
        return
    report = {
        "status": "blocked",
        "reason": "RLDS image streaming is intentionally not reimplemented in this host-only script. Run the Docker-backed extractor extension with tensorflow_datasets available.",
        "requested_limit": args.limit,
        "checkpoint": str(args.checkpoint),
        "timestamp": time.time(),
    }
    write_json(args.output_root / "reports" / "feature_cache_smoke.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
