#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()


COLORS = {
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
    parser.add_argument("--output-subdir", default="visualizations_world_topdown")
    parser.add_argument("--graph-kind", choices=["world_graph", "observable_graph"], default="world_graph")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def node_map(graph: dict) -> dict[str, dict]:
    return {str(node["id"]): node for node in graph.get("nodes", [])}


def diagnostics_positions(diagnostics: dict) -> dict[str, tuple[float, float, float]]:
    positions = diagnostics.get("world_positions", {})
    out: dict[str, tuple[float, float, float]] = {}
    for node_id, xyz in positions.items():
        out[str(node_id)] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return out


def world_xy(node_id: str, positions: dict[str, tuple[float, float, float]]) -> tuple[float, float] | None:
    xyz = positions.get(node_id)
    if xyz is None:
        return None
    return float(xyz[0]), float(xyz[1])


def bounds(position_maps: list[dict[str, tuple[float, float, float]]]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for positions in position_maps:
        for node_id in positions:
            point = world_xy(node_id, positions)
            if point is None:
                continue
            xs.append(point[0])
            ys.append(point[1])
    if not xs:
        return -0.5, 0.5, -0.5, 0.5
    pad_x = max(0.05, (max(xs) - min(xs)) * 0.12)
    pad_y = max(0.05, (max(ys) - min(ys)) * 0.12)
    return min(xs) - pad_x, max(xs) + pad_x, min(ys) - pad_y, max(ys) + pad_y


def project(point: tuple[float, float], extent: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = extent
    x, y = point
    sx = (x - min_x) / max(1e-9, max_x - min_x)
    sy = (y - min_y) / max(1e-9, max_y - min_y)
    return 36 + sx * (width - 72), height - 36 - sy * (height - 72)


def draw_arrow(draw, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int], width: int = 2) -> None:
    import math

    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = max(1.0, math.hypot(dx, dy))
    pad = 15.0
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


def draw_frame(graph: dict, positions: dict[str, tuple[float, float, float]], extent: tuple[float, float, float, float], out_path: Path) -> None:
    from PIL import Image, ImageDraw

    panel_w, panel_h = 540, 540
    legend_w = 380
    img = Image.new("RGB", (panel_w + legend_w, panel_h), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    nodes = node_map(graph)
    node_ids = sorted(nodes)
    node_numbers = {node_id: idx + 1 for idx, node_id in enumerate(node_ids)}
    anchors = {
        node_id: project(point, extent, panel_w, panel_h)
        for node_id in nodes
        if (point := world_xy(node_id, positions)) is not None
    }

    draw.rectangle((0, 0, panel_w, panel_h), fill=(248, 248, 248, 255))
    draw.line((36, panel_h - 36, panel_w - 24, panel_h - 36), fill=(120, 120, 120, 255), width=1)
    draw.line((36, panel_h - 36, 36, 24), fill=(120, 120, 120, 255), width=1)
    draw.text((panel_w - 48, panel_h - 28), "+X", fill=(0, 0, 0, 255))
    draw.text((12, 22), "+Y", fill=(0, 0, 0, 255))

    priority = {"left_of": 0, "right_of": 0, "above": 1, "below": 1, "inside": 2, "contains": 2, "on": 3, "grasping": 4}
    for edge in sorted(graph.get("binary_edges", []), key=lambda item: (priority.get(item["predicate"], 9), item["subject"], item["object"])):
        subject = edge["subject"]
        obj = edge["object"]
        predicate = edge["predicate"]
        if subject not in anchors or obj not in anchors:
            continue
        color = COLORS.get(predicate, (70, 70, 70))
        alpha = 240 if predicate in {"on", "inside", "contains", "grasping"} else 150
        draw_arrow(draw, anchors[subject], anchors[obj], (*color, alpha), width=4 if predicate == "grasping" else 2)

    for node_id, (x, y) in sorted(anchors.items(), key=lambda item: node_numbers[item[0]]):
        node = nodes[node_id]
        visible = bool(node.get("visible", False))
        fill = (255, 255, 255, 245) if visible else (210, 210, 210, 235)
        outline = (20, 20, 20, 255)
        radius = 12 if node_id == "gripper" else 10
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
        label = str(node_numbers[node_id])
        draw.text((x - 4, y - 7), label, fill=(0, 0, 0, 255))

    lx = panel_w + 16
    y = 14
    draw.text((lx, y), f"frame {graph['frame_id']} / rule-based world graph", fill=(0, 0, 0, 255))
    y += 24
    draw.text((lx, y), "Node number: id  diagnostic xyz", fill=(0, 0, 0, 255))
    y += 18
    for node_id in node_ids:
        node = nodes[node_id]
        xyz = positions.get(node_id)
        xyz_text = "none" if xyz is None else f"{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}"
        visibility = "" if node.get("visible") else " hidden"
        draw.text((lx, y), f"{node_numbers[node_id]}: {node_id}{visibility}", fill=(0, 0, 0, 255))
        y += 15
        draw.text((lx + 16, y), xyz_text, fill=(70, 70, 70, 255))
        y += 15
    y += 10
    counts = Counter(edge["predicate"] for edge in graph.get("binary_edges", []))
    draw.text((lx, y), f"Edges: {len(graph.get('binary_edges', []))}", fill=(0, 0, 0, 255))
    y += 20
    for predicate in ["left_of", "right_of", "above", "below", "on", "grasping", "inside", "contains"]:
        if counts[predicate] <= 0:
            continue
        color = COLORS[predicate]
        draw.line((lx, y + 7, lx + 30, y + 7), fill=color, width=4 if predicate == "grasping" else 2)
        draw.text((lx + 38, y), f"{predicate}: {counts[predicate]}", fill=(0, 0, 0, 255))
        y += 18

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    args = parse_args()
    selected = load_json(args.output_dir / "reports" / "selected_frames.json")["selected_frames"]
    graph_dir = args.output_dir / "rule_based" / args.graph_kind / args.demo_id
    diagnostics_dir = args.output_dir / "diagnostics" / args.demo_id
    all_positions = [diagnostics_positions(load_json(path)) for path in sorted(diagnostics_dir.glob("*.json"))]
    extent = bounds(all_positions)
    out_dir = args.output_dir / args.output_subdir / args.demo_id
    frames = []
    for item in selected:
        frame_id = int(item["frame_id"])
        graph = load_json(graph_dir / f"{frame_id:06d}.json")
        diagnostics = load_json(diagnostics_dir / f"{frame_id:06d}.json")
        draw_frame(graph, diagnostics_positions(diagnostics), extent, out_dir / f"{frame_id:06d}.png")
        frames.append(frame_id)
    print(json.dumps({"visualization_dir": str(out_dir), "graph_kind": args.graph_kind, "frames": frames}, indent=2))


if __name__ == "__main__":
    main()
