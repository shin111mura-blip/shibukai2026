#!/usr/bin/env python3
"""Visualize cached BBox-derived scene graphs without importing OpenVLA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


PAIRWISE_COLORS = {
    "left_of": (0, 135, 255),
    "right_of": (0, 90, 220),
    "above": (0, 170, 120),
    "below": (0, 130, 90),
    "near": (255, 170, 0),
    "overlap": (220, 40, 40),
}


def load_jsonl_by_image_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rows[row["image_id"]] = row
    return rows


def font(size: int = 13) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def resolve_image_path(path_text: str, workspace_root: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    if path_text.startswith("/workspace/"):
        mapped = workspace_root / path_text.removeprefix("/workspace/")
        if mapped.exists():
            return mapped
    if path_text.startswith("/workspace/HRI2027/"):
        mapped = workspace_root / path_text.removeprefix("/workspace/HRI2027/")
        if mapped.exists():
            return mapped
    return path


def node_center_px(node: dict[str, Any], width: int, height: int) -> tuple[float, float]:
    cx, cy = node["center"]
    return float(cx) * width, float(cy) * height


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    fill: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
    text_font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=text_font, stroke_width=0)
    pad = 2
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    draw.text((x, y), text, fill=fill, font=text_font)


def draw_header(draw: ImageDraw.ImageDraw, meta: dict[str, Any], graph: dict[str, Any], width: int) -> None:
    text_font = font(12)
    lines = [
        f"task_{int(meta['task_id']):02d} demo={meta['demo_id']} step={meta['timestep']}",
        str(meta.get("task_name", ""))[:110],
        f"nodes={len(graph.get('nodes', []))} pairwise={len(graph.get('pairwise_edges', []))} between={len(graph.get('between_hyperedges', []))}",
    ]
    y = 4
    for line in lines:
        draw_label(draw, (4, y), line, text_font=text_font)
        y += 17


def draw_bboxes(image: Image.Image, graph: dict[str, Any], meta: dict[str, Any]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    text_font = font(13)
    draw_header(draw, meta, graph, width)
    for node in graph.get("nodes", []):
        x1, y1, x2, y2 = [float(v) for v in node["bbox"]]
        box = (x1 * width, y1 * height, x2 * width, y2 * height)
        draw.rectangle(box, outline=(255, 80, 0), width=2)
        label = f"{node['node_id']}:{node['category']}[{node['instance_id']}] {float(node['confidence']):.2f}"
        draw_label(draw, (box[0] + 2, box[1] + 2), label, text_font=text_font)
    return out


def draw_pairwise(image: Image.Image, graph: dict[str, Any], meta: dict[str, Any], relation_filter: Iterable[str] | None) -> Image.Image:
    out = draw_bboxes(image, graph, meta)
    draw = ImageDraw.Draw(out)
    width, height = out.size
    text_font = font(11)
    nodes = {int(node["node_id"]): node for node in graph.get("nodes", [])}
    keep = set(relation_filter) if relation_filter else None
    for edge in graph.get("pairwise_edges", []):
        rels = [rel for rel in edge.get("relations", []) if keep is None or rel in keep]
        if not rels:
            continue
        src = nodes.get(int(edge["source"]))
        dst = nodes.get(int(edge["target"]))
        if src is None or dst is None:
            continue
        p1 = node_center_px(src, width, height)
        p2 = node_center_px(dst, width, height)
        color = PAIRWISE_COLORS.get(rels[0], (0, 180, 255))
        draw.line((p1, p2), fill=color, width=2)
        mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
        draw_label(draw, (mx, my), ",".join(rels), fill=color, text_font=text_font)
    return out


def draw_between(image: Image.Image, graph: dict[str, Any], meta: dict[str, Any]) -> Image.Image:
    out = draw_bboxes(image, graph, meta)
    draw = ImageDraw.Draw(out)
    width, height = out.size
    text_font = font(11)
    nodes = {int(node["node_id"]): node for node in graph.get("nodes", [])}
    for edge in graph.get("between_hyperedges", []):
        target = nodes.get(int(edge["target"]))
        ref1 = nodes.get(int(edge["reference_1"]))
        ref2 = nodes.get(int(edge["reference_2"]))
        if target is None or ref1 is None or ref2 is None:
            continue
        points = (node_center_px(ref1, width, height), node_center_px(target, width, height), node_center_px(ref2, width, height))
        draw.line(points, fill=(255, 0, 180), width=3)
        draw_label(draw, points[1], f"between({edge['target']},{edge['reference_1']},{edge['reference_2']})", fill=(255, 0, 180), text_font=text_font)
    return out


def draw_network(graph: dict[str, Any], meta: dict[str, Any], size: int = 900) -> Image.Image:
    out = Image.new("RGB", (size, size), (248, 248, 248))
    draw = ImageDraw.Draw(out)
    text_font = font(14)
    small_font = font(11)
    draw_header(draw, meta, graph, size)
    nodes = graph.get("nodes", [])
    if not nodes:
        draw_label(draw, (20, size // 2), "no nodes", text_font=text_font)
        return out
    center = (size / 2, size / 2 + 30)
    radius = min(size * 0.36, 300)
    positions: dict[int, tuple[float, float]] = {}
    for idx, node in enumerate(nodes):
        angle = 2 * math.pi * idx / max(1, len(nodes)) - math.pi / 2
        positions[int(node["node_id"])] = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
    for edge in graph.get("pairwise_edges", []):
        src = positions.get(int(edge["source"]))
        dst = positions.get(int(edge["target"]))
        if src is None or dst is None:
            continue
        rels = edge.get("relations", [])
        color = PAIRWISE_COLORS.get(rels[0], (90, 90, 90)) if rels else (90, 90, 90)
        draw.line((src, dst), fill=color, width=1)
    for edge in graph.get("between_hyperedges", []):
        pts = [positions.get(int(edge[key])) for key in ("reference_1", "target", "reference_2")]
        if all(point is not None for point in pts):
            draw.line(tuple(pts), fill=(255, 0, 180), width=3)
    for node in nodes:
        node_id = int(node["node_id"])
        x, y = positions[node_id]
        r = 18
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw.text((x - 5, y - 8), str(node_id), fill=(0, 0, 0), font=text_font)
        label = f"{node_id}:{node['category']}[{node['instance_id']}]"
        draw_label(draw, (x - 45, y + 22), label, text_font=small_font)
    return out


def select_graphs(
    manifest: dict[str, dict[str, Any]],
    graphs: dict[str, dict[str, Any]],
    max_images_per_task: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    counts: dict[int, int] = {}
    selected = []
    for image_id, meta in manifest.items():
        graph = graphs.get(image_id)
        if graph is None:
            continue
        task_id = int(meta["task_id"])
        if counts.get(task_id, 0) >= max_images_per_task:
            continue
        selected.append((meta, graph))
        counts[task_id] = counts.get(task_id, 0) + 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-graph-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_visualization"))
    parser.add_argument("--max-images-per-task", type=int, default=20)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--pairwise-relations", nargs="*", default=["left_of", "right_of", "above", "below", "near", "overlap"])
    args = parser.parse_args()

    manifest = load_jsonl_by_image_id(args.manifest)
    graphs = load_jsonl_by_image_id(args.scene_graph_cache)
    selected = select_graphs(manifest, graphs, args.max_images_per_task)

    summary: dict[str, Any] = {
        "manifest": str(args.manifest),
        "scene_graph_cache": str(args.scene_graph_cache),
        "selected_images": len(selected),
        "max_images_per_task": args.max_images_per_task,
        "tasks": {},
    }
    for meta, graph in selected:
        image_path = resolve_image_path(meta["image_path"], args.workspace_root)
        image = Image.open(image_path).convert("RGB")
        task_id = int(meta["task_id"])
        task_dir = args.output_dir / f"task_{task_id:02d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        safe_id = graph["image_id"].replace("/", "__")
        draw_bboxes(image, graph, meta).save(task_dir / f"{safe_id}_bbox.png")
        draw_pairwise(image, graph, meta, args.pairwise_relations).save(task_dir / f"{safe_id}_pairwise.png")
        draw_between(image, graph, meta).save(task_dir / f"{safe_id}_between.png")
        draw_network(graph, meta).save(task_dir / f"{safe_id}_network.png")
        task_summary = summary["tasks"].setdefault(str(task_id), {"count": 0, "nodes": [], "pairwise_edges": [], "between_hyperedges": []})
        task_summary["count"] += 1
        task_summary["nodes"].append(len(graph.get("nodes", [])))
        task_summary["pairwise_edges"].append(len(graph.get("pairwise_edges", [])))
        task_summary["between_hyperedges"].append(len(graph.get("between_hyperedges", [])))

    for task_summary in summary["tasks"].values():
        for key in ("nodes", "pairwise_edges", "between_hyperedges"):
            values = task_summary[key]
            task_summary[f"{key}_min"] = min(values) if values else 0
            task_summary[f"{key}_max"] = max(values) if values else 0
            task_summary[f"{key}_mean"] = sum(values) / len(values) if values else 0.0
            del task_summary[key]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} scene graph samples to {args.output_dir}")


if __name__ == "__main__":
    main()
