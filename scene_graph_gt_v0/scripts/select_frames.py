#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()


def edge_key(graph: dict) -> tuple:
    return tuple((e["subject"], e["predicate"], e["object"]) for e in graph.get("binary_edges", []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-id", default="demo_0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_gt_v0"))
    parser.add_argument("--max-selected", type=int, default=20)
    parser.add_argument("--graph-kind", choices=["world_graph", "observable_graph"], default="world_graph")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_dir = args.output_dir / "rule_based" / args.graph_kind / args.demo_id
    files = sorted(graph_dir.glob("*.json"))
    selected = {}
    priority = []
    previous = None
    previous_predicates = set()
    for idx, path in enumerate(files):
        graph = json.loads(path.read_text(encoding="utf-8"))
        reasons = []
        if idx == 0:
            reasons.append("initial_frame")
            priority.append(int(path.stem))
        if idx == len(files) - 1:
            reasons.append("final_frame")
            priority.append(int(path.stem))
        current = edge_key(graph)
        if previous is not None and current != previous:
            reasons.append("graph_changed")
            priority.append(int(path.stem))
        predicates = {edge["predicate"] for edge in graph.get("binary_edges", [])}
        for predicate in ("grasping", "on", "inside"):
            if predicate in predicates and predicate not in previous_predicates:
                reasons.append(f"{predicate}_start")
                priority.append(int(path.stem))
            if predicate not in predicates and predicate in previous_predicates:
                reasons.append(f"{predicate}_end")
                priority.append(int(path.stem))
        if reasons:
            selected[int(path.stem)] = sorted(set(reasons))
        previous = current
        previous_predicates = predicates
    if files:
        selected.setdefault(int(files[-1].stem), []).append("final_frame")
    selected_order = []
    for frame in priority:
        if frame not in selected_order:
            selected_order.append(frame)
        if len(selected_order) >= args.max_selected:
            break
    if len(selected_order) < args.max_selected and files:
        step = max(1, len(files) // args.max_selected)
        for path in files[::step]:
            selected.setdefault(int(path.stem), []).append("uniform_interval")
            frame = int(path.stem)
            if frame not in selected_order:
                selected_order.append(frame)
            if len(selected_order) >= args.max_selected:
                break
    if files and int(files[-1].stem) not in selected_order:
        if len(selected_order) >= args.max_selected:
            selected_order[-1] = int(files[-1].stem)
        else:
            selected_order.append(int(files[-1].stem))
    selected_items = [{"frame_id": frame, "reasons": sorted(set(selected.get(frame, [])))} for frame in sorted(selected_order)]
    payload = {"demo_id": args.demo_id, "graph_kind": args.graph_kind, "selected_frames": selected_items}
    out = args.output_dir / "reports" / "selected_frames.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
