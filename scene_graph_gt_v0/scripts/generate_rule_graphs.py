#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from scene_graph.rule_generator import resolve_task, run_demo


TARGET = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-language", default=TARGET)
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_gt_v0"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = resolve_task(args.suite, args.task_language)
    summary = run_demo(
        task=task,
        demo_index=args.demo_index,
        output_dir=args.output_dir,
        image_size=args.image_size,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
