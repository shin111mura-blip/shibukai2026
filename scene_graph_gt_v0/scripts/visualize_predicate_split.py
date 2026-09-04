#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from visualize_comparison import PREDICATE_COLORS, anchor_points, draw_arrow, edge_tuple, node_map


GROUPS = [
    ("horizontal", {"left_of", "right_of"}),
    ("vertical", {"above", "below"}),
    ("on", {"on"}),
    ("grasping", {"grasping"}),
    ("inside/contains", {"inside", "contains"}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-id", default="demo_0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_gt_v0"))
    parser.add_argument("--output-subdir", default="visualizations_rot180_split")
    parser.add_argument("--rotate-180", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def draw_group_panel(base_img, graph: dict, anchors: dict[str, tuple[float, float]], title: str, predicates: set[str], node_numbers: dict[str, int]):
    from PIL import ImageDraw

    img = base_img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    edges = [edge_tuple(edge) for edge in graph.get("binary_edges", []) if edge.get("predicate") in predicates]
    for subject, predicate, obj in sorted(edges):
        if subject not in anchors or obj not in anchors:
            continue
        color = PREDICATE_COLORS.get(predicate, (80, 80, 80))
        width = 5 if predicate == "grasping" else 3
        draw_arrow(draw, anchors[subject], anchors[obj], (*color, 235), width=width)
    for node_id, (x, y) in sorted(anchors.items()):
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(255, 255, 255, 235), outline=(0, 0, 0, 255), width=2)
        draw.text((x - 3, y - 6), str(node_numbers.get(node_id, "?")), fill=(0, 0, 0, 255))
    draw.rectangle((0, 0, img.width, 24), fill=(255, 255, 255, 225))
    draw.text((8, 5), f"{title}  edges={len(edges)}", fill=(0, 0, 0, 255))
    return img


def legend_panel(width: int, height: int, frame_id: int, node_numbers: dict[str, int], unanchored_nodes: set[str]) -> "Image.Image":
    from PIL import Image, ImageDraw

    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    y = 10
    draw.text((10, y), f"frame {frame_id} / rotated 180 deg", fill=(0, 0, 0))
    y += 24
    draw.text((10, y), "Nodes", fill=(0, 0, 0))
    y += 18
    for node_id, idx in sorted(node_numbers.items(), key=lambda item: item[1]):
        suffix = "  Qwen-only" if node_id in unanchored_nodes else ""
        draw.text((14, y), f"{idx}: {node_id}{suffix}", fill=(0, 0, 0))
        y += 15
    y += 10
    draw.text((10, y), "Colors", fill=(0, 0, 0))
    y += 18
    for predicate in ["left_of/right_of", "above/below", "on", "grasping", "inside/contains"]:
        key = predicate.split("/")[0]
        color = PREDICATE_COLORS.get(key, (80, 80, 80))
        draw.line((14, y + 7, 44, y + 7), fill=color, width=5 if key == "grasping" else 3)
        draw.text((52, y), predicate, fill=(0, 0, 0))
        y += 18
    return panel


def main() -> None:
    from PIL import Image

    args = parse_args()
    selected = json.loads((args.output_dir / "reports" / "selected_frames.json").read_text(encoding="utf-8"))["selected_frames"]
    out_dir = args.output_dir / args.output_subdir / args.demo_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        frame_id = int(item["frame_id"])
        base = Image.open(args.output_dir / "frames" / args.demo_id / f"{frame_id:06d}.png").convert("RGB")
        if args.rotate_180:
            base = base.transpose(Image.Transpose.ROTATE_180)
        rule = json.loads((args.output_dir / "rule_based" / "observable_graph" / args.demo_id / f"{frame_id:06d}.json").read_text(encoding="utf-8"))
        qwen_path = args.output_dir / "qwen_zero_shot" / "parsed" / args.demo_id / f"{frame_id:06d}.json"
        qwen = json.loads(qwen_path.read_text(encoding="utf-8")) if qwen_path.exists() else {"nodes": [], "binary_edges": []}

        node_ids = sorted(set(node_map(rule)) | set(node_map(qwen)))
        node_numbers = {node_id: idx + 1 for idx, node_id in enumerate(node_ids)}
        rule_anchors = anchor_points(rule, base.width, base.height, args.rotate_180)
        rule_node_ids = set(rule_anchors)
        qwen_anchors = {node["id"]: rule_anchors[node["id"]] for node in qwen.get("nodes", []) if node.get("id") in rule_anchors}
        qwen_only_unanchored = set(node_map(qwen)) - rule_node_ids

        active_groups = []
        all_predicates = {edge.get("predicate") for edge in rule.get("binary_edges", []) + qwen.get("binary_edges", [])}
        for group_name, predicates in GROUPS:
            if predicates & all_predicates:
                active_groups.append((group_name, predicates))
        if not active_groups:
            active_groups = GROUPS[:1]

        panel_w, panel_h = base.size
        legend_w = 360
        canvas = Image.new("RGB", (panel_w * 2 + legend_w, panel_h * len(active_groups)), "white")
        for row, (group_name, predicates) in enumerate(active_groups):
            y = row * panel_h
            canvas.paste(draw_group_panel(base, rule, rule_anchors, f"Rule / {group_name}", predicates, node_numbers), (0, y))
            canvas.paste(draw_group_panel(base, qwen, qwen_anchors, f"Qwen / {group_name}", predicates, node_numbers), (panel_w, y))
            if row == 0:
                canvas.paste(legend_panel(legend_w, panel_h, frame_id, node_numbers, qwen_only_unanchored), (panel_w * 2, y))
        canvas.save(out_dir / f"{frame_id:06d}.png")
    print(json.dumps({"visualization_dir": str(out_dir), "frames": [item["frame_id"] for item in selected]}, indent=2))


if __name__ == "__main__":
    main()
