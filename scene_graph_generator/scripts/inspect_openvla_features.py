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
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/openvla_7b_base_with_libero_spatial_10demo_stats"))
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--feature-layer", type=int, default=-2)
    args = ap.parse_args()
    try:
        import torch
        import transformers
    except Exception as exc:
        report = {
            "status": "blocked",
            "reason": f"Missing OpenVLA runtime dependency: {exc}",
            "checkpoint": str(args.checkpoint),
            "feature_layer": args.feature_layer,
        }
        write_json(args.output_root / "reports" / "openvla_feature_inspection.json", report)
        (args.output_root / "reports" / "openvla_feature_inspection.md").write_text(
            "# OpenVLA Feature Inspection\n\nBlocked: torch/transformers are not available in this Python environment.\n"
        )
        print(json.dumps(report, sort_keys=True))
        return
    report = {
        "status": "runtime_available",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "checkpoint": str(args.checkpoint),
        "note": "Use cache_openvla_features.py --limit 1 inside the Docker/runtime environment to run the actual 1-frame forward.",
    }
    write_json(args.output_root / "reports" / "openvla_feature_inspection.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
