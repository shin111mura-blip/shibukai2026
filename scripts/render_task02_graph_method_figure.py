#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def node_positions(graph: dict) -> dict[str, list[float]]:
    positions: dict[str, list[float]] = {}
    for node in graph.get("nodes", []):
        xyz = node.get("position_world_xyz")
        if xyz is not None and len(xyz) == 3:
            positions[str(node["id"])] = [float(v) for v in xyz]
    return positions


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


def common_bounds(graphs: list[dict]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    pts: list[list[float]] = []
    for graph in graphs:
        pts.extend(node_positions(graph).values())
    if not pts:
        raise ValueError("No nodes with position_world_xyz were found.")
    xs, ys, zs = zip(*pts)
    radius = max(max(max(v) - min(v) for v in (xs, ys, zs)) * 0.62, 0.16)
    centers = [(max(v) + min(v)) * 0.5 for v in (xs, ys, zs)]
    return (
        (centers[0] - radius, centers[0] + radius),
        (centers[1] - radius, centers[1] + radius),
        (centers[2] - radius * 0.50, centers[2] + radius * 0.80),
    )


def style_axis(ax, bounds) -> None:
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_zlim(*bounds[2])
    ax.view_init(elev=24, azim=-55)
    ax.set_axis_off()
    ax.grid(False)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.65))
    except Exception:
        pass
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_alpha(0.0)
            axis.line.set_alpha(0.0)
        except Exception:
            pass


def draw_clean_graph(ax, graph: dict, bounds, *, node_size: float, edge_width: float, edge_alpha: float, edge_limit: int) -> None:
    positions = node_positions(graph)
    edges = [
        edge
        for edge in graph.get("binary_edges", [])
        if edge.get("subject") in positions and edge.get("object") in positions
    ]
    grasp_edges = [edge for edge in edges if edge.get("predicate") == "grasping"]
    other_edges = [edge for edge in edges if edge.get("predicate") != "grasping"]
    edges = grasp_edges + other_edges[: max(0, edge_limit - len(grasp_edges))]

    for edge in edges:
        a = positions[str(edge["subject"])]
        b = positions[str(edge["object"])]
        predicate = str(edge["predicate"])
        width = edge_width * (1.45 if predicate == "grasping" else 1.0)
        alpha = min(1.0, edge_alpha + 0.15) if predicate == "grasping" else edge_alpha
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color=relation_color(predicate),
            linewidth=width,
            alpha=alpha,
            solid_capstyle="round",
        )

    for node_id, xyz in positions.items():
        ax.scatter(
            [xyz[0]],
            [xyz[1]],
            [xyz[2]],
            s=node_size,
            c="#2f6fbd",
            marker="o",
            edgecolors="#0f2742",
            linewidths=1.4,
            depthshade=False,
        )
    style_axis(ax, bounds)


def save_all(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render clean Task02 GraphGenerator method figures.")
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path(
            "outputs/rollout_depthfree_graph_generator_cached_base_openvla_xyz_v2_visualizations/gt/test/"
            "02_regular_task_02_high_official_libero_spatial_a809a2a68be345f5_frame_000004.json"
        ),
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=Path(
            "outputs/rollout_depthfree_graph_generator_cached_base_openvla_xyz_v2_visualizations/predictions/test/"
            "02_regular_task_02_high_official_libero_spatial_a809a2a68be345f5_frame_000004.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_method_figures/task02_graph_3d_circles_png"),
    )
    parser.add_argument("--node-size", type=float, default=360.0)
    parser.add_argument("--edge-width", type=float, default=3.2)
    parser.add_argument("--edge-alpha", type=float, default=0.62)
    parser.add_argument("--edge-limit", type=int, default=80)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = read_json(args.gt)
    pred = read_json(args.pred)
    bounds = common_bounds([gt, pred])

    fig = plt.figure(figsize=(10, 4.4), dpi=260)
    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
    draw_clean_graph(ax_gt, gt, bounds, node_size=args.node_size, edge_width=args.edge_width, edge_alpha=args.edge_alpha, edge_limit=args.edge_limit)
    draw_clean_graph(ax_pred, pred, bounds, node_size=args.node_size, edge_width=args.edge_width, edge_alpha=args.edge_alpha, edge_limit=args.edge_limit)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.02)
    save_all(fig, args.output_dir / "task02_gt_pred_large_nodes_no_text")
    plt.close(fig)

    for name, graph in (("task02_gt_large_nodes_no_text", gt), ("task02_pred_large_nodes_no_text", pred)):
        single = plt.figure(figsize=(5.0, 4.4), dpi=260)
        ax = single.add_subplot(1, 1, 1, projection="3d")
        draw_clean_graph(ax, graph, bounds, node_size=args.node_size, edge_width=args.edge_width, edge_alpha=args.edge_alpha, edge_limit=args.edge_limit)
        single.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        save_all(single, args.output_dir / name)
        plt.close(single)

    print(
        {
            "output_dir": str(args.output_dir),
            "pair_png": str(args.output_dir / "task02_gt_pred_large_nodes_no_text.png"),
            "gt_png": str(args.output_dir / "task02_gt_large_nodes_no_text.png"),
            "pred_png": str(args.output_dir / "task02_pred_large_nodes_no_text.png"),
        }
    )


if __name__ == "__main__":
    main()
