#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .common import DEFAULT_OUTPUT_ROOT
except ImportError:  # pragma: no cover - CLI path execution
    from common import DEFAULT_OUTPUT_ROOT
from scene_graph_generator.graph_generator.schema import write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--architecture", choices=["pooled_mlp", "node_query_decoder"], default="node_query_decoder")
    args = ap.parse_args()
    try:
        import torch  # noqa: F401
    except Exception as exc:
        report = {
            "status": "blocked",
            "architecture": args.architecture,
            "reason": f"torch is required for Graph Generator training: {exc}",
        }
        write_json(args.output_root / "reports" / f"{args.architecture}_training_status.json", report)
        print(json.dumps(report, sort_keys=True))
        return
    report = {
        "status": "blocked",
        "architecture": args.architecture,
        "reason": "Feature cache shards are required before training. Run cache_openvla_features.py in an OpenVLA runtime first.",
    }
    write_json(args.output_root / "reports" / f"{args.architecture}_training_status.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
