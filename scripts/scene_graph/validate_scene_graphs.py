#!/usr/bin/env python3
"""Validate and summarize generated oracle scene graph JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from oracle_scene_graph_utils import read_jsonl, safe_json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-dir", type=Path, default=Path("outputs/scene_graph_probe/graphs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_probe/diagnostics"))
    parser.add_argument("--overlay-dir", type=Path, default=Path("outputs/scene_graph_probe/overlays"))
    return parser.parse_args()


def write_counter_csv(path: Path, header: List[str], counter: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for key, value in sorted(counter.items()):
            writer.writerow([key, value])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    relation_counts: Counter = Counter()
    contact_counts: Counter = Counter()
    category_counts: Counter = Counter()
    missing_fields: Counter = Counter()
    episode_timesteps: Dict[str, int] = defaultdict(int)
    no_relation_episodes = set()
    seen_episodes = set()
    failures: List[Dict[str, Any]] = []
    total_nodes = 0
    total_edges = 0
    total_contacts = 0
    empty_graphs = 0
    small_object_graphs = 0
    timestep_count = 0
    grasping_count = 0
    suite = None
    task_id = None
    task_name = None

    required_fields = ["suite", "task_id", "episode_id", "timestep", "nodes", "edges", "contacts", "metadata"]
    for graph_path in sorted(args.graphs_dir.glob("episode_*.jsonl")):
        episode_has_relation = False
        for line_idx, record in enumerate(read_jsonl(graph_path), start=1):
            timestep_count += 1
            episode_key = str(record.get("episode_id", graph_path.stem))
            seen_episodes.add(episode_key)
            episode_timesteps[episode_key] += 1
            suite = suite or record.get("suite")
            task_id = task_id if task_id is not None else record.get("task_id")
            task_name = task_name or record.get("task_name")
            for field in required_fields:
                if field not in record:
                    missing_fields[field] += 1
                    failures.append({"file": str(graph_path), "line": line_idx, "missing_field": field})
            nodes = record.get("nodes", [])
            edges = record.get("edges", [])
            contacts = record.get("contacts", [])
            total_nodes += len(nodes)
            total_edges += len(edges)
            total_contacts += len(contacts)
            object_count = sum(1 for node in nodes if node.get("type") == "object")
            if not nodes:
                empty_graphs += 1
            if object_count <= 1:
                small_object_graphs += 1
            if edges:
                episode_has_relation = True
            for node in nodes:
                category_counts[node.get("category", "unknown")] += 1
            for edge in edges:
                rel = edge.get("rel", "unknown")
                relation_counts[rel] += 1
                if rel == "grasping":
                    grasping_count += 1
            for contact in contacts:
                key = f"{contact.get('geom1')}|{contact.get('geom2')}"
                contact_counts[key] += 1
        if not episode_has_relation:
            no_relation_episodes.add(graph_path.stem)

    avg_nodes = total_nodes / timestep_count if timestep_count else 0.0
    avg_edges = total_edges / timestep_count if timestep_count else 0.0
    summary = {
        "suite": suite,
        "task_id": task_id,
        "task_name": task_name,
        "episode_count": len(seen_episodes),
        "timestep_count": timestep_count,
        "avg_node_count": avg_nodes,
        "avg_edge_count": avg_edges,
        "relation_counts": dict(relation_counts),
        "category_counts": dict(category_counts),
        "contact_count": total_contacts,
        "grasping_count": grasping_count,
        "empty_graph_ratio": empty_graphs / timestep_count if timestep_count else None,
        "object_leq_one_ratio": small_object_graphs / timestep_count if timestep_count else None,
        "no_relation_episodes": sorted(no_relation_episodes),
        "missing_fields": dict(missing_fields),
        "episode_timesteps": dict(episode_timesteps),
    }

    write_counter_csv(args.output_dir / "relation_distribution.csv", ["relation", "count"], relation_counts)
    write_counter_csv(args.output_dir / "contact_distribution.csv", ["contact_pair", "count"], contact_counts)
    safe_json_dump(summary, args.output_dir / "graph_generation_summary.json")
    safe_json_dump(failures, args.output_dir / "failure_cases.json")

    overlays = sorted(args.overlay_dir.glob("episode_*/t*_overlay.png"))
    report_lines = [
        "# Oracle Scene Graph Generation Report",
        "",
        f"- Suite / task: `{suite}` / `{task_id}`",
        f"- Task name: `{task_name}`",
        f"- Episodes: {len(seen_episodes)}",
        f"- Timesteps: {timestep_count}",
        f"- Average nodes: {avg_nodes:.2f}",
        f"- Average edges: {avg_edges:.2f}",
        f"- Contact count: {total_contacts}",
        f"- Grasping count: {grasping_count}",
        f"- Empty graph ratio: {summary['empty_graph_ratio']}",
        f"- Object <= 1 ratio: {summary['object_leq_one_ratio']}",
        "",
        "## Retrieved Information",
        "",
        "- Object/body/geom/site names are retrieved through MuJoCo model APIs when available.",
        "- Object and gripper world positions are retrieved from `body_xpos`, `geom_xpos`, or `site_xpos`.",
        "- Contact pairs are retrieved from `sim.data.contact` / `sim.data.ncon` when available.",
        "",
        "## Object Node Extraction",
        "",
        "LIBERO object registries are preferred, with MuJoCo body/geom name heuristics as fallback. Robot/table/floor/wall/region names are filtered.",
        "",
        "## Relation Rules",
        "",
        "- `next_to`: XY distance threshold.",
        "- `between`: distance to segment, projection ratio, endpoint distance, and pair distance thresholds.",
        "- `on`: object above another object with XY proximity and contact.",
        "- `inside`: container-category proximity approximation.",
        "- `touching`: MuJoCo contact pairs mapped to object nodes.",
        "- `grasping`: gripper-object contact plus gripper distance candidate.",
        "",
        "## Relation Distribution",
        "",
    ]
    if relation_counts:
        report_lines.extend(f"- {rel}: {count}" for rel, count in sorted(relation_counts.items()))
    else:
        report_lines.append("- none")
    report_lines.extend(
        [
            "",
            "## Overlay",
            "",
            f"- Generated overlay count detected: {len(overlays)}",
            f"- Representative overlay: `{overlays[0] if overlays else 'not generated'}`",
            "",
            "## Current Limits",
            "",
            "- Geometry extents are not yet used for exact `on` / `inside` reasoning.",
            "- RGB overlay falls back to graph-only XY layout when rollout RGB images were not saved.",
            "- Camera projection is inspected separately in `env_inspection/camera_info.json` and is not required for graph JSONL generation.",
            "",
        ]
    )
    (args.output_dir / "graph_generation_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    failure_lines = ["# Failure Cases", ""]
    if failures:
        failure_lines.extend(f"- {json.dumps(item, sort_keys=True)}" for item in failures[:200])
    else:
        failure_lines.append("No missing required fields were detected.")
    if no_relation_episodes:
        failure_lines.extend(["", "## Episodes With No Relations", ""])
        failure_lines.extend(f"- {episode}" for episode in sorted(no_relation_episodes))
    (args.output_dir / "failure_cases.md").write_text("\n".join(failure_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
