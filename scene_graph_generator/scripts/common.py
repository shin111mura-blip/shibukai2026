from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph_generator.graph_generator.feature_cache import read_jsonl, write_jsonl


DEFAULT_GRAPH_ROOT = Path("outputs/scene_graph_gt_openvla_spatial/rule_based/world_graph")
DEFAULT_DEMO_MANIFEST = Path("outputs/scene_graph_gt_openvla_spatial/reports/openvla_demo_manifest.jsonl")
DEFAULT_OUTPUT_ROOT = Path("outputs/scene_graph_generator_openvla_spatial")


def load_demo_manifest(path: Path = DEFAULT_DEMO_MANIFEST) -> Dict[int, Dict]:
    rows = read_jsonl(path)
    return {int(row["global_episode_index"]): row for row in rows}


def write_md_table(path: Path, headers: List[str], rows: List[List]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    path.write_text("\n".join(lines) + "\n")

