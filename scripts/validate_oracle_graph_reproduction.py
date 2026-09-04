#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rollout_collection_common import REPORTS, ROOT, TEACHER_GRAPH_3D_ROOT, WORLD_GRAPH_ROOT, read_json, sha256_payload, write_json


def canonical_triplets(graph: dict) -> list[tuple[str, str, str]]:
    return sorted((e["subject"], e["predicate"], e["object"]) for e in graph.get("binary_edges", []))


def canonical_nodes(graph: dict) -> list[str]:
    return sorted(n["id"] for n in graph.get("nodes", []))


def compare_pair(world_path: Path, teacher_path: Path) -> dict:
    world = read_json(world_path)
    teacher = read_json(teacher_path)
    world_triplets = canonical_triplets(world)
    teacher_triplets = canonical_triplets(teacher)
    world_nodes = canonical_nodes(world)
    teacher_nodes = canonical_nodes(teacher)
    return {
        "world_path": str(world_path.relative_to(ROOT)),
        "teacher_path": str(teacher_path.relative_to(ROOT)),
        "node_exact_match": world_nodes == teacher_nodes,
        "triplet_exact_match": world_triplets == teacher_triplets,
        "world_triplet_count": len(world_triplets),
        "teacher_triplet_count": len(teacher_triplets),
        "world_payload_sha256": sha256_payload({"nodes": world_nodes, "triplets": world_triplets}),
        "teacher_payload_sha256": sha256_payload({"nodes": teacher_nodes, "triplets": teacher_triplets}),
        "world_only_triplets": [list(x) for x in sorted(set(world_triplets) - set(teacher_triplets))[:10]],
        "teacher_only_triplets": [list(x) for x in sorted(set(teacher_triplets) - set(world_triplets))[:10]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare existing oracle graph targets against 3D teacher graph copies.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--world-root", type=Path, default=WORLD_GRAPH_ROOT)
    parser.add_argument("--teacher-root", type=Path, default=TEACHER_GRAPH_3D_ROOT)
    args = parser.parse_args()
    pairs = []
    for world_path in sorted(args.world_root.glob("task_*/*/*.json")):
        rel = world_path.relative_to(args.world_root)
        teacher_path = args.teacher_root / rel
        if teacher_path.exists():
            pairs.append((world_path, teacher_path))
        if len(pairs) >= args.limit:
            break
    results = [compare_pair(w, t) for w, t in pairs]
    node_matches = sum(r["node_exact_match"] for r in results)
    triplet_matches = sum(r["triplet_exact_match"] for r in results)
    relation_counts = {}
    for r in results:
        for path_key in ["world_path"]:
            graph = read_json(ROOT / r[path_key])
            for edge in graph.get("binary_edges", []):
                pred = edge["predicate"]
                relation_counts[pred] = relation_counts.get(pred, 0) + 1
    summary = {
        "status": "partial_static_equivalence_only",
        "note": "This compares saved world graph targets with saved teacher_graph_3d copies. It is not a live simulator-state rerun.",
        "samples_compared": len(results),
        "node_exact_match_rate": node_matches / len(results) if results else 0.0,
        "triplet_exact_match_rate": triplet_matches / len(results) if results else 0.0,
        "mismatch_count": sum(not r["triplet_exact_match"] or not r["node_exact_match"] for r in results),
        "relation_counts": relation_counts,
        "examples": results[:5],
        "mismatches": [r for r in results if not r["triplet_exact_match"] or not r["node_exact_match"]][:10],
    }
    write_json(REPORTS / "oracle_graph_reproduction_static.json", summary)
    lines = [
        "# Oracle Graph Reproduction Check",
        "",
        "Status: PARTIAL STATIC CHECK ONLY.",
        "",
        "This check compares existing saved `rule_based/world_graph` targets with saved `teacher_graph_3d/world_graph` copies for 100 frames. It does not rerun the simulator state, because Full Collection is blocked before rollout/preflight by unresolved policy checkpoints.",
        "",
        f"- samples compared: `{summary['samples_compared']}`",
        f"- node exact match rate: `{summary['node_exact_match_rate']:.6f}`",
        f"- triplet exact match rate: `{summary['triplet_exact_match_rate']:.6f}`",
        f"- mismatch count: `{summary['mismatch_count']}`",
        f"- relation counts: `{summary['relation_counts']}`",
        "",
        "Live simulator-state rerun remains required before Full Collection can start.",
    ]
    (REPORTS / "oracle_graph_reproduction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(REPORTS / "oracle_graph_reproduction.md"))
    raise SystemExit(0 if summary["mismatch_count"] == 0 and summary["samples_compared"] == args.limit else 1)


if __name__ == "__main__":
    main()
