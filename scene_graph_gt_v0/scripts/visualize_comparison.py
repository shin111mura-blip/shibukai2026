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


PREDICATE_COLORS = {
    "left_of": (40, 125, 255),
    "right_of": (40, 125, 255),
    "above": (34, 150, 84),
    "below": (34, 150, 84),
    "on": (235, 126, 35),
    "inside": (132, 86, 190),
    "contains": (132, 86, 190),
    "grasping": (220, 55, 90),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-id", default="demo_0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_gt_v0"))
    parser.add_argument("--output-subdir", default="visualizations_rot180")
    parser.add_argument("--rotate-180", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-edge-labels", type=int, default=28)
    parser.add_argument("--style", choices=["minimal", "labeled"], default="minimal")
    return parser.parse_args()


def edge_tuple(edge: dict) -> tuple[str, str, str]:
    return str(edge["subject"]), str(edge["predicate"]), str(edge["object"])


def edge_set(graph: dict) -> set[tuple[str, str, str]]:
    return {edge_tuple(edge) for edge in graph.get("binary_edges", [])}


def node_map(graph: dict) -> dict[str, dict]:
    return {str(node["id"]): node for node in graph.get("nodes", [])}


def rotated_point(point: tuple[float, float] | None, width: int, height: int, rotate: bool) -> tuple[float, float] | None:
    if point is None:
        return None
    x, y = point
    if not rotate:
        return x, y
    return float(width - 1 - x), float(height - 1 - y)


def anchor_points(graph: dict, width: int, height: int, rotate: bool) -> dict[str, tuple[float, float]]:
    nodes = node_map(graph)
    anchors: dict[str, tuple[float, float]] = {}
    missing: list[str] = []
    for node_id, node in sorted(nodes.items()):
        point = node.get("centroid_xy")
        if point is not None:
            anchors[node_id] = rotated_point((float(point[0]), float(point[1])), width, height, rotate)  # type: ignore[assignment]
        else:
            missing.append(node_id)
    if missing:
        # Keep nodes without image centroids visible in a deterministic side arc.
        radius = min(width, height) * 0.34
        center = (width * 0.50, height * 0.50)
        for idx, node_id in enumerate(missing):
            theta = -math.pi / 2 + 2 * math.pi * idx / max(1, len(missing))
            anchors[node_id] = (center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta))
    return anchors


def draw_arrow(draw, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int], width: int = 2) -> None:
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = max(1.0, math.hypot(dx, dy))
    # Pull endpoints inward so arrows do not cover node dots.
    pad = 13.0
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


def draw_graph_overlay(
    base_img,
    graph: dict,
    anchors: dict[str, tuple[float, float]],
    title: str,
    max_edge_labels: int,
    node_numbers: dict[str, int],
    style: str,
):
    from PIL import ImageDraw

    img = base_img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    edges = [edge_tuple(edge) for edge in graph.get("binary_edges", [])]
    # Draw lower-priority spatial relations first, grasping/on last.
    priority = {"left_of": 0, "right_of": 0, "above": 1, "below": 1, "inside": 2, "contains": 2, "on": 3, "grasping": 4}
    for subject, predicate, obj in sorted(edges, key=lambda item: (priority.get(item[1], 9), item)):
        if subject not in anchors or obj not in anchors:
            continue
        color = PREDICATE_COLORS.get(predicate, (80, 80, 80))
        alpha = 230 if predicate in {"grasping", "on", "inside", "contains"} else 95
        draw_arrow(draw, anchors[subject], anchors[obj], (*color, alpha), width=4 if predicate == "grasping" else 2)
    for node_id, (x, y) in sorted(anchors.items()):
        node = node_map(graph).get(node_id, {})
        visible = bool(node.get("visible", True))
        fill = (255, 255, 255, 235) if visible else (180, 180, 180, 210)
        outline = (20, 20, 20, 255)
        radius = 9
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
        label = str(node_numbers.get(node_id, "?"))
        draw.text((x - 3, y - 6), label, fill=(0, 0, 0, 255))
        if style == "labeled":
            full_label = node_id.replace("_", " ")
            draw.rectangle((x + 11, y - 9, x + 11 + min(220, 7 * len(full_label)), y + 7), fill=(255, 255, 255, 210))
            draw.text((x + 14, y - 9), full_label, fill=(0, 0, 0, 255))
    draw.rectangle((0, 0, img.width, 24), fill=(255, 255, 255, 220))
    draw.text((8, 5), title, fill=(0, 0, 0, 255))

    counts = Counter(predicate for _s, predicate, _o in edges)
    if style == "labeled":
        legend_y = 30
        legend = ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))
        for line in textwrap.wrap(legend, width=72)[:3]:
            draw.rectangle((6, legend_y - 2, img.width - 6, legend_y + 15), fill=(255, 255, 255, 190))
            draw.text((10, legend_y), line, fill=(0, 0, 0, 255))
            legend_y += 17
    else:
        x0, y0 = 8, 31
        for predicate in ["left_of", "right_of", "above", "below", "on", "grasping"]:
            if counts[predicate] <= 0:
                continue
            color = PREDICATE_COLORS[predicate]
            draw.line((x0, y0 + 6, x0 + 22, y0 + 6), fill=(*color, 230), width=4 if predicate == "grasping" else 2)
            draw.text((x0 + 26, y0), f"{predicate}:{counts[predicate]}", fill=(0, 0, 0, 255))
            y0 += 15

    if style == "labeled":
        label_count = 0
        for subject, predicate, obj in sorted(edges):
            if label_count >= max_edge_labels:
                break
            if subject not in anchors or obj not in anchors:
                continue
            if predicate not in {"grasping", "on", "inside", "contains"} and label_count > max_edge_labels // 2:
                continue
            sx, sy = anchors[subject]
            ox, oy = anchors[obj]
            mx, my = (sx + ox) / 2, (sy + oy) / 2
            label = predicate
            color = PREDICATE_COLORS.get(predicate, (80, 80, 80))
            draw.rectangle((mx - 3, my - 8, mx + 7 * len(label) + 3, my + 8), fill=(255, 255, 255, 170))
            draw.text((mx, my - 8), label, fill=(*color, 255))
            label_count += 1
    return img


def text_panel(
    width: int,
    height: int,
    frame_id: int,
    rule: dict,
    qwen: dict,
    node_numbers: dict[str, int],
    unanchored_nodes: set[str],
) -> "Image.Image":
    from PIL import Image, ImageDraw

    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    rule_edges = edge_set(rule)
    qwen_edges = edge_set(qwen)
    reverse_nodes = {v: k for k, v in node_numbers.items()}
    lines = [
        f"frame {frame_id} / RGB rotated 180 deg",
        "",
        "Nodes:",
    ]
    lines.extend(
        f"{idx}: {reverse_nodes[idx]}{'  Qwen-only/no anchor' if reverse_nodes[idx] in unanchored_nodes else ''}"
        for idx in sorted(reverse_nodes)
    )
    lines.extend([
        "",
        "Edge color:",
        "blue=L/R, green=above/below, orange=on, red=grasping",
        "",
        f"Rule nodes: {len(rule.get('nodes', []))}  edges: {len(rule_edges)}",
        f"Qwen nodes: {len(qwen.get('nodes', []))}  edges: {len(qwen_edges)}",
        f"match: {len(rule_edges & qwen_edges)}",
        f"rule-only: {len(rule_edges - qwen_edges)}",
        f"qwen-only: {len(qwen_edges - rule_edges)}",
        "",
        "Matched edges:",
    ])
    lines.extend(f"  {node_numbers.get(s,'?')} -{p}-> {node_numbers.get(o,'?')}" for s, p, o in sorted(rule_edges & qwen_edges)[:18])
    lines.append("")
    lines.append("Qwen-only edges:")
    lines.extend(f"  {node_numbers.get(s,'?')} -{p}-> {node_numbers.get(o,'?')}" for s, p, o in sorted(qwen_edges - rule_edges)[:24])
    y = 10
    for raw_line in lines:
        for line in textwrap.wrap(raw_line, width=62) or [""]:
            draw.text((10, y), line, fill=(20, 20, 20))
            y += 14
            if y > height - 20:
                return panel
    return panel


def write_summary(path: Path, frame_id: int, rule: dict, qwen: dict, anchor_source: str, rotate: bool) -> None:
    payload = {
        "frame_id": frame_id,
        "rotated_180": rotate,
        "qwen_anchor_source": anchor_source,
        "rule_nodes": rule.get("nodes", []),
        "rule_edges": rule.get("binary_edges", []),
        "qwen_nodes": qwen.get("nodes", []),
        "qwen_edges": qwen.get("binary_edges", []),
        "matched_edges": [list(edge) for edge in sorted(edge_set(rule) & edge_set(qwen))],
        "rule_only_edges": [list(edge) for edge in sorted(edge_set(rule) - edge_set(qwen))],
        "qwen_only_edges": [list(edge) for edge in sorted(edge_set(qwen) - edge_set(rule))],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    from PIL import Image

    args = parse_args()
    selected = json.loads((args.output_dir / "reports" / "selected_frames.json").read_text(encoding="utf-8"))["selected_frames"]
    out_dir = args.output_dir / args.output_subdir / args.demo_id
    summary_dir = args.output_dir / args.output_subdir / "summaries" / args.demo_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        frame_id = int(item["frame_id"])
        img = Image.open(args.output_dir / "frames" / args.demo_id / f"{frame_id:06d}.png").convert("RGB")
        if args.rotate_180:
            img = img.transpose(Image.Transpose.ROTATE_180)
        rule = json.loads((args.output_dir / "rule_based" / "observable_graph" / args.demo_id / f"{frame_id:06d}.json").read_text(encoding="utf-8"))
        pred_path = args.output_dir / "qwen_zero_shot" / "parsed" / args.demo_id / f"{frame_id:06d}.json"
        qwen = json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.exists() else {"nodes": [], "binary_edges": []}

        rule_anchors = anchor_points(rule, img.width, img.height, args.rotate_180)
        node_ids = sorted(set(node_map(rule)) | set(node_map(qwen)))
        node_numbers = {node_id: idx + 1 for idx, node_id in enumerate(node_ids)}
        # Qwen graphs are vision-only and do not carry centroids. Use the rule
        # segmentation centroids only for node ids grounded by the rule graph.
        qwen_anchors = {node["id"]: rule_anchors[node["id"]] for node in qwen.get("nodes", []) if node.get("id") in rule_anchors}
        qwen_only_unanchored = set(node_map(qwen)) - set(rule_anchors)

        rule_panel = draw_graph_overlay(img, rule, rule_anchors, "Rule-based observable graph", args.max_edge_labels, node_numbers, args.style)
        qwen_panel = draw_graph_overlay(img, qwen, qwen_anchors, "Qwen3-VL vision-only graph", args.max_edge_labels, node_numbers, args.style)
        info_panel = text_panel(max(360, img.width), img.height, frame_id, rule, qwen, node_numbers, qwen_only_unanchored)

        canvas = Image.new("RGB", (rule_panel.width + qwen_panel.width + info_panel.width, img.height), "white")
        canvas.paste(rule_panel, (0, 0))
        canvas.paste(qwen_panel, (rule_panel.width, 0))
        canvas.paste(info_panel, (rule_panel.width + qwen_panel.width, 0))
        canvas.save(out_dir / f"{frame_id:06d}.png")
        write_summary(summary_dir / f"{frame_id:06d}.json", frame_id, rule, qwen, "rule_based_centroid_xy_only", args.rotate_180)

    print(
        json.dumps(
            {
                "visualization_dir": str(out_dir),
                "summary_dir": str(summary_dir),
                "rotated_180": args.rotate_180,
                "frames": [item["frame_id"] for item in selected],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
