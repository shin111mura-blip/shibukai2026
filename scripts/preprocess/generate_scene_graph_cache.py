#!/usr/bin/env python3
"""Generate Scene Graph JSONL cache from a YOLO BBox cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from prismatic.vla.scene_graph import RelationThresholds, build_scene_graph, scene_graph_hash


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("cache/scene_graph"))
    parser.add_argument("--near-threshold", type=float, default=0.25)
    parser.add_argument("--left-right-margin", type=float, default=0.02)
    parser.add_argument("--above-below-margin", type=float, default=0.02)
    parser.add_argument("--overlap-iou-threshold", type=float, default=0.05)
    parser.add_argument("--between-perpendicular-threshold", type=float, default=0.10)
    parser.add_argument("--between-endpoint-margin", type=float, default=0.05)
    args = parser.parse_args()

    thresholds = RelationThresholds(
        near_threshold=args.near_threshold,
        left_right_margin=args.left_right_margin,
        above_below_margin=args.above_below_margin,
        overlap_iou_threshold=args.overlap_iou_threshold,
        between_perpendicular_threshold=args.between_perpendicular_threshold,
        between_endpoint_margin=args.between_endpoint_margin,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "libero_spatial_scene_graph.jsonl"
    bbox_hash = file_hash(args.bbox_cache)
    count = 0
    with open(args.bbox_cache) as f, open(out_path, "w") as out:
        for line in f:
            row = json.loads(line)
            sg = build_scene_graph(
                row["image_id"],
                row.get("detections", []),
                thresholds=thresholds,
                source_bbox_cache_id=bbox_hash,
            )
            payload = sg.to_dict()
            payload["scene_graph_hash"] = scene_graph_hash(sg)
            out.write(json.dumps(payload) + "\n")
            count += 1
    metadata = {
        "source_bbox_cache": str(args.bbox_cache),
        "source_bbox_cache_sha256": bbox_hash,
        "threshold_config": thresholds.to_dict(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_graphs": count,
    }
    with open(args.output_dir / "scene_graph_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {count} scene graphs to {out_path}")


if __name__ == "__main__":
    main()
