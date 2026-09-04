#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import DEFAULT_GRAPH_ROOT, DEFAULT_OUTPUT_ROOT
except ImportError:  # pragma: no cover - CLI path execution
    from common import DEFAULT_GRAPH_ROOT, DEFAULT_OUTPUT_ROOT
from scene_graph_generator.graph_generator.ontology import build_ontology, save_ontology


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = ap.parse_args()
    ontology = build_ontology(args.graph_root)
    save_ontology(ontology, args.output_root / "ontology")
    print(f"K={len(ontology['nodes'])} R={len(ontology['predicates'])} hash={ontology['ontology_hash']}")


if __name__ == "__main__":
    main()
