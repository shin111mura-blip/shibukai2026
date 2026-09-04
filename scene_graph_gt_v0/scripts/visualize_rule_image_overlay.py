#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import Counter
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()


COLORS = {
    "left_of": (40, 125, 255),
    "right_of": (40, 125, 255),
    "above": (34, 150, 84),
    "below": (34, 150, 84),
    "front_of": (0, 150, 140),
    "behind": (0, 150, 140),
    "on": (235, 126, 35),
    "inside": (132, 86, 190),
    "contains": (132, 86, 190),
    "grasping": (220, 55, 90),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-id", default="demo_0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_gt_v0"))
    parser.add_argument("--output-subdir", default="visualizations_rule_image_overlay")
    parser.add_argument("--graph-kind", choices=["world_graph", "observable_graph"], default="world_graph")
    parser.add_argument("--rotate-180", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def node_map(graph: dict) -> dict[str, dict]:
    return {str(node["id"]): node for node in graph.get("nodes", [])}


def rotated_point(point: tuple[float, float], width: int, height: int, rotate: bool) -> tuple[float, float]:
    if not rotate:
        return point
    x, y = point
    return float(width - 1 - x), float(height - 1 - y)


def anchors_from_diagnostics(diagnostics: dict, width: int, height: int, rotate: bool) -> dict[str, dict]:
    anchors: dict[str, dict] = {}
    visibility = diagnostics.get("visibility", {}).get("node_visibility", {})
    for node_id, payload in visibility.items():
        if str(node_id) == "gripper":
            point = payload.get("centroid_xy")
            source = "robot_mask"
        else:
            point = payload.get("segmentation_centroid_xy")
            source = "segmentation"
            if point is None:
                point = payload.get("centroid_xy")
                source = "world_projection"
        if point is None:
            continue
        anchors[str(node_id)] = {
            "point": rotated_point((float(point[0]), float(point[1])), width, height, rotate),
            "source": source,
        }
    return anchors


def draw_arrow(draw, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int, int], width: int = 2) -> None:
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = max(1.0, math.hypot(dx, dy))
    pad = 14.0
    sx += dx / length * pad
    sy += dy / length * pad
    ex -= dx / length * pad
    ey -= dy / length * pad
    draw.line((sx, sy, ex, ey), fill=color, width=width)
    angle = math.atan2(ey - sy, ex - sx)
    head = 8
    for offset in (2.55, -2.55):
        hx = ex + head * math.cos(angle + offset)
        hy = ey + head * math.sin(angle + offset)
        draw.line((ex, ey, hx, hy), fill=color, width=width)


def draw_overlay(base_img, graph: dict, anchors: dict[str, dict], node_numbers: dict[str, int], title: str):
    from PIL import ImageDraw

    img = base_img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    priority = {"left_of": 0, "right_of": 0, "above": 1, "below": 1, "front_of": 2, "behind": 2, "inside": 3, "contains": 3, "on": 4, "grasping": 5}
    for edge in sorted(graph.get("binary_edges", []), key=lambda item: (priority.get(item["predicate"], 9), item["subject"], item["object"])):
        subject = edge["subject"]
        obj = edge["object"]
        predicate = edge["predicate"]
        if subject not in anchors or obj not in anchors:
            continue
        color = COLORS.get(predicate, (80, 80, 80))
        projected = anchors[subject]["source"] == "world_projection" or anchors[obj]["source"] == "world_projection"
        alpha = 215 if predicate in {"on", "inside", "contains", "grasping"} else 120
        if projected:
            alpha = min(alpha, 90)
        draw_arrow(draw, anchors[subject]["point"], anchors[obj]["point"], (*color, alpha), width=4 if predicate == "grasping" else 2)

    ordered_anchors = sorted(
        anchors.items(),
        key=lambda item: (item[1]["source"] == "world_projection", node_numbers.get(item[0], 999)),
    )
    for node_id, anchor in ordered_anchors:
        if node_id not in node_numbers:
            continue
        x, y = anchor["point"]
        radius = 12 if node_id == "gripper" else 10
        if anchor["source"] == "world_projection":
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 242, 170, 175), outline=(150, 80, 20, 245), width=3)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 255, 255, 235), outline=(0, 0, 0, 255), width=2)
        label = str(node_numbers[node_id])
        draw.text((x - 4, y - 7), label, fill=(0, 0, 0, 255))

    draw.rectangle((0, 0, img.width, 24), fill=(255, 255, 255, 225))
    draw.text((8, 5), title, fill=(0, 0, 0, 255))
    return img


def legend_panel(width: int, height: int, frame_id: int, graph: dict, node_numbers: dict[str, int], anchors: dict[str, dict]) -> "Image.Image":
    from PIL import Image, ImageDraw

    nodes = node_map(graph)
    reverse_nodes = {idx: node_id for node_id, idx in node_numbers.items()}
    counts = Counter(edge["predicate"] for edge in graph.get("binary_edges", []))
    lines: list[tuple[str, tuple[int, int, int]]] = []
    lines.append((f"frame {frame_id} / rule-based {graph.get('mode')} graph", (0, 0, 0)))
    lines.append(("", (0, 0, 0)))
    lines.append(("Nodes", (0, 0, 0)))
    for idx in sorted(reverse_nodes):
        node_id = reverse_nodes[idx]
        category = nodes.get(node_id, {}).get("category", "")
        line = f"{idx}: {node_id}"
        if category and category != node_id:
            line += f"  [{category}]"
        if node_id not in anchors:
            line += "  no image anchor"
        elif anchors[node_id]["source"] == "world_projection":
            line += "  projected"
        for wrapped in textwrap.wrap(line, width=64) or [""]:
            lines.append((wrapped, (0, 0, 0)))
    lines.append(("", (0, 0, 0)))
    lines.append((f"Edges: {len(graph.get('binary_edges', []))}", (0, 0, 0)))
    for predicate in ["left_of", "right_of", "above", "below", "front_of", "behind", "on", "grasping", "inside", "contains"]:
        if counts[predicate] <= 0:
            continue
        lines.append((f"{predicate}: {counts[predicate]}", COLORS[predicate]))
    lines.append(("", (0, 0, 0)))
    lines.append(("Edge list", (0, 0, 0)))
    number_by_node = {node_id: idx for node_id, idx in node_numbers.items()}
    for edge in sorted(graph.get("binary_edges", []), key=lambda item: (item["predicate"], item["subject"], item["object"])):
        subject = number_by_node.get(edge["subject"], edge["subject"])
        obj = number_by_node.get(edge["object"], edge["object"])
        line = f"{subject} -{edge['predicate']}-> {obj}"
        lines.append((line, COLORS.get(edge["predicate"], (0, 0, 0))))
    lines.append(("", (0, 0, 0)))
    note = "Solid nodes use image segmentation. Pale nodes are MuJoCo-projected anchors for graph nodes that are currently occluded. Graph nodes still do not contain coordinates or hidden flags."
    for wrapped in textwrap.wrap(note, width=48):
        lines.append((wrapped, (70, 70, 70)))

    line_height = 15
    panel_height = max(height, 24 + line_height * len(lines))
    panel = Image.new("RGB", (width, panel_height), "white")
    draw = ImageDraw.Draw(panel)
    y = 12
    for text, color in lines:
        if text:
            draw.text((12, y), text, fill=color)
        y += line_height
    return panel


def main() -> None:
    from PIL import Image

    args = parse_args()
    selected = load_json(args.output_dir / "reports" / "selected_frames.json")["selected_frames"]
    graph_dir = args.output_dir / "rule_based" / args.graph_kind / args.demo_id
    diagnostics_dir = args.output_dir / "diagnostics" / args.demo_id
    frame_dir = args.output_dir / "frames" / args.demo_id
    out_dir = args.output_dir / args.output_subdir / args.demo_id
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[int] = []
    for item in selected:
        frame_id = int(item["frame_id"])
        graph = load_json(graph_dir / f"{frame_id:06d}.json")
        diagnostics = load_json(diagnostics_dir / f"{frame_id:06d}.json")
        image = Image.open(frame_dir / f"{frame_id:06d}.png").convert("RGB")
        if args.rotate_180:
            image = image.transpose(Image.Transpose.ROTATE_180)
        anchors = anchors_from_diagnostics(diagnostics, image.width, image.height, args.rotate_180)
        node_ids = sorted(node["id"] for node in graph.get("nodes", []))
        node_numbers = {node_id: idx + 1 for idx, node_id in enumerate(node_ids)}
        overlay = draw_overlay(image, graph, anchors, node_numbers, f"Rule graph on RGB / frame {frame_id}")
        legend = legend_panel(520, image.height, frame_id, graph, node_numbers, anchors)
        canvas = Image.new("RGB", (overlay.width + legend.width, max(overlay.height, legend.height)), "white")
        canvas.paste(overlay, (0, 0))
        canvas.paste(legend, (overlay.width, 0))
        canvas.save(out_dir / f"{frame_id:06d}.png")
        frames.append(frame_id)
    print(json.dumps({"visualization_dir": str(out_dir), "graph_kind": args.graph_kind, "rotated_180": args.rotate_180, "frames": frames}, indent=2))


if __name__ == "__main__":
    main()
