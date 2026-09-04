from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def estimate_feature_bytes(num_frames: int, tokens_per_frame: int, hidden_dim: int, bytes_per_value: int = 2) -> int:
    # features + bool attention mask + int64 token type mask, ignoring safetensors metadata overhead.
    return num_frames * (tokens_per_frame * hidden_dim * bytes_per_value + tokens_per_frame * (1 + 8))

