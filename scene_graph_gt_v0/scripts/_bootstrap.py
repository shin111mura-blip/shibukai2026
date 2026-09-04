from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> Path:
    root = Path(__file__).resolve().parents[2]
    package_root = root / "scene_graph_gt_v0"
    for path in (root, package_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root

