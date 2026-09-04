#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from rollout_collection_common import DATA_ROOT, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create episode-level rollout train/validation/test splits.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    rows = read_jsonl(args.data_root / "manifests" / "all.jsonl")
    groups = defaultdict(list)
    for row in rows:
        key = (row.get("suite_name", "libero_spatial"), int(row["task_id"]), int(row.get("initial_state_id", -1)))
        groups[key].append(row)
    keys = list(groups)
    random.Random(args.seed).shuffle(keys)
    n = len(keys)
    split_for_key = {}
    for i, key in enumerate(keys):
        frac = i / max(n, 1)
        split_for_key[key] = "train" if frac < 0.70 else ("validation" if frac < 0.85 else "test")
    out = {"train": [], "validation": [], "test": []}
    for key, group_rows in groups.items():
        out[split_for_key[key]].extend(group_rows)
    for split, split_rows in out.items():
        write_jsonl(args.data_root / "manifests" / f"{split}.jsonl", split_rows)
    leakage = []
    seen = {}
    for split, split_rows in out.items():
        for row in split_rows:
            key = (row.get("suite_name", "libero_spatial"), int(row["task_id"]), int(row.get("initial_state_id", -1)))
            if key in seen and seen[key] != split:
                leakage.append({"group": key, "splits": [seen[key], split]})
            seen[key] = split
    write_json(args.data_root / "reports" / "split_summary.json", {k: len(v) for k, v in out.items()} | {"initial_state_leakage": leakage, "seed": args.seed})
    print({k: len(v) for k, v in out.items()})
    raise SystemExit(0 if not leakage else 1)


if __name__ == "__main__":
    main()
