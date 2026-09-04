#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .common import DEFAULT_OUTPUT_ROOT
except ImportError:  # pragma: no cover
    from common import DEFAULT_OUTPUT_ROOT

from scene_graph_generator.graph_generator.feature_cache import read_jsonl
from scene_graph_generator.graph_generator.schema import graph_triplets, read_json, write_json


def graph_path(root: Path, graph_root_name: str, row: dict) -> Path:
    return (
        root
        / graph_root_name
        / "world_graph"
        / f"task_{row['task_id']:02d}"
        / f"global_{row['global_episode_index']:06d}"
        / f"{row['frame_index']:06d}.json"
    )


def pred_path(root: Path, arch: str, split: str, row: dict) -> Path:
    return root / "predictions" / arch / split / f"task_{row['task_id']:02d}" / f"global_{row['global_episode_index']:06d}" / f"{row['frame_index']:06d}.json"


def node_positions(graph: dict) -> dict[str, list[float]]:
    out = {}
    for node in graph.get("nodes", []):
        xyz = node.get("position_world_xyz")
        if xyz is not None and len(xyz) == 3:
            out[node["id"]] = [float(x) for x in xyz]
    return out


def relation_color(predicate: str) -> str:
    return {
        "grasping": "#d62728",
        "on": "#2ca02c",
        "above": "#9467bd",
        "below": "#8c564b",
        "left_of": "#1f77b4",
        "right_of": "#17becf",
        "front_of": "#ff7f0e",
        "behind": "#bcbd22",
    }.get(predicate, "#555555")


def common_bounds(graphs: list[dict]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    pts = []
    for graph in graphs:
        pts.extend(node_positions(graph).values())
    if not pts:
        return None
    xs, ys, zs = zip(*pts)
    ranges = [max(v) - min(v) for v in (xs, ys, zs)]
    radius = max(max(ranges) * 0.58, 0.12)
    centers = [(max(v) + min(v)) * 0.5 for v in (xs, ys, zs)]
    return (
        (centers[0] - radius, centers[0] + radius),
        (centers[1] - radius, centers[1] + radius),
        (centers[2] - radius * 0.45, centers[2] + radius * 0.75),
    )


def coordinate_errors(gt: dict, pred: dict) -> dict:
    import math

    gt_pos = node_positions(gt)
    pred_pos = node_positions(pred)
    per_node = {}
    for node_id in sorted(set(gt_pos) & set(pred_pos)):
        delta = [pred_pos[node_id][i] - gt_pos[node_id][i] for i in range(3)]
        l2 = math.sqrt(sum(x * x for x in delta))
        per_node[node_id] = {
            "gt": gt_pos[node_id],
            "pred": pred_pos[node_id],
            "delta_xyz": delta,
            "l2_m": l2,
        }
    l2_values = [row["l2_m"] for row in per_node.values()]
    return {
        "per_node": per_node,
        "mean_l2_m": sum(l2_values) / len(l2_values) if l2_values else None,
        "max_l2_m": max(l2_values) if l2_values else None,
    }


def draw_graph_3d(ax, graph: dict, title: str, *, edge_limit: int = 40, bounds=None) -> None:
    positions = node_positions(graph)
    edges = [edge for edge in graph.get("binary_edges", []) if edge.get("subject") in positions and edge.get("object") in positions]
    grasp_edges = [e for e in edges if e.get("predicate") == "grasping"]
    other_edges = [e for e in edges if e.get("predicate") != "grasping"]
    edges = grasp_edges + other_edges[: max(0, edge_limit - len(grasp_edges))]

    for node_id, xyz in positions.items():
        color = "#111111" if node_id == "gripper" else "#4477aa"
        marker = "^" if node_id == "gripper" else "o"
        ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], s=62, c=color, marker=marker, depthshade=True)
        ax.text(xyz[0], xyz[1], xyz[2], " " + node_id.replace("_", "\n", 1), fontsize=7)

    for edge in edges:
        a = positions[edge["subject"]]
        b = positions[edge["object"]]
        predicate = edge["predicate"]
        color = relation_color(predicate)
        width = 2.5 if predicate == "grasping" else 0.9
        alpha = 0.95 if predicate == "grasping" else 0.35
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=width, alpha=alpha)
        mid = [(a[i] + b[i]) * 0.5 for i in range(3)]
        if predicate in {"grasping", "on"}:
            ax.text(mid[0], mid[1], mid[2], predicate, fontsize=7, color=color)

    if bounds is not None:
        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])
        ax.set_zlim(*bounds[2])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=-55)


def render_pair(root: Path, arch: str, row: dict, out_path: Path, *, graph_root_name: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = read_json(graph_path(root, graph_root_name, row))
    pred = read_json(pred_path(root, arch, row["split"], row))

    fig = plt.figure(figsize=(13, 6), dpi=160)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    bounds = common_bounds([gt, pred])
    draw_graph_3d(ax1, gt, "GT 3D Scene Graph", bounds=bounds)
    draw_graph_3d(ax2, pred, "Predicted 3D Scene Graph", bounds=bounds)
    fig.suptitle(
        f"split={row['split']} task={row['task_id']:02d} global={row['global_episode_index']:06d} frame={row['frame_index']:06d}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    gt_edges = set(graph_triplets(gt))
    pred_edges = set(graph_triplets(pred))
    errors = coordinate_errors(gt, pred)
    return {
        "image": str(out_path),
        "row": {k: row[k] for k in ("split", "task_id", "global_episode_index", "frame_index", "sample_key")},
        "gt_edges": len(gt_edges),
        "pred_edges": len(pred_edges),
        "tp_edges": len(gt_edges & pred_edges),
        "fp_edges": len(pred_edges - gt_edges),
        "fn_edges": len(gt_edges - pred_edges),
        "gt_grasping": sorted([e for e in gt_edges if e[1] == "grasping"]),
        "pred_grasping": sorted([e for e in pred_edges if e[1] == "grasping"]),
        "coordinate_errors": errors,
    }


def select_rows(root: Path, arch: str, splits: set[str], *, graph_root_name: str, max_examples: int) -> list[tuple[str, dict]]:
    rows = [r for r in read_jsonl(root / "feature_cache" / "all_frames" / "cache_manifest.jsonl") if r["split"] in splits]
    by_ep = defaultdict(list)
    for row in rows:
        pp = pred_path(root, arch, row["split"], row)
        gp = graph_path(root, graph_root_name, row)
        if pp.exists() and gp.exists():
            by_ep[(row["split"], row["task_id"], row["global_episode_index"])].append(row)
    for key in by_ep:
        by_ep[key].sort(key=lambda r: r["frame_index"])

    selected: list[tuple[str, dict]] = []
    used = set()
    for _key, ep_rows in sorted(by_ep.items()):
        for idx, row in enumerate(ep_rows):
            gt_edges = set(graph_triplets(read_json(graph_path(root, graph_root_name, row))))
            if any(edge[1] == "grasping" for edge in gt_edges):
                for j in range(max(0, idx - 1), min(len(ep_rows), idx + 2)):
                    sample_key = ep_rows[j]["sample_key"]
                    if sample_key not in used:
                        selected.append(("grasp", ep_rows[j]))
                        used.add(sample_key)
                break
        if len([x for x in selected if x[0] == "grasp"]) >= max_examples // 2:
            break

    temporal_candidates = []
    for key, ep_rows in by_ep.items():
        prev_edges = None
        for row in ep_rows:
            edges = set(graph_triplets(read_json(graph_path(root, graph_root_name, row))))
            if prev_edges is not None:
                change = len(edges - prev_edges) + len(prev_edges - edges)
                if change:
                    temporal_candidates.append((change, key, row))
            prev_edges = edges
    for change, _key, row in sorted(
        temporal_candidates,
        key=lambda x: (x[0], x[1][1], x[1][2], x[2]["frame_index"]),
        reverse=True,
    ):
        if len(selected) >= max_examples:
            break
        if row["sample_key"] in used:
            continue
        selected.append((f"temporal_change_{change}", row))
        used.add(row["sample_key"])

    if len(selected) < max_examples:
        for _key, ep_rows in sorted(by_ep.items()):
            for row in ep_rows:
                if len(selected) >= max_examples:
                    break
                if row["sample_key"] not in used:
                    selected.append(("regular", row))
                    used.add(row["sample_key"])
            if len(selected) >= max_examples:
                break
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--architecture", default="pooled_mlp_depth_3d")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--graph-root-name", default="teacher_graph_3d")
    ap.add_argument("--max-examples", type=int, default=12)
    args = ap.parse_args()

    out_dir = args.output_root / "visualizations" / args.architecture / "graph_3d"
    selected = select_rows(
        args.output_root,
        args.architecture,
        {x.strip() for x in args.splits.split(",") if x.strip()},
        graph_root_name=args.graph_root_name,
        max_examples=args.max_examples,
    )
    examples = []
    for idx, (kind, row) in enumerate(selected):
        out_path = out_dir / f"{idx:02d}_{kind}_task_{row['task_id']:02d}_global_{row['global_episode_index']:06d}_frame_{row['frame_index']:06d}.png"
        item = render_pair(args.output_root, args.architecture, row, out_path, graph_root_name=args.graph_root_name)
        item["kind"] = kind
        examples.append(item)
    report = {"status": "ok", "architecture": args.architecture, "dir": str(out_dir), "examples": examples}
    write_json(out_dir / "graph_3d_manifest.json", report)
    print(json.dumps({"status": "ok", "dir": str(out_dir), "count": len(examples)}, sort_keys=True))


if __name__ == "__main__":
    main()
