#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rollout_collection_common import DATA_ROOT, read_json, validate_episode_dir, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge worker-completed rollout episode metadata into all.jsonl.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    rows = []
    for episode_dir in sorted((args.data_root / "episodes").glob("*/*/*")):
        if not episode_dir.is_dir():
            continue
        ok, errors, meta = validate_episode_dir(episode_dir)
        if ok:
            meta["episode_dir"] = str(episode_dir)
            rows.append(meta)
    write_jsonl(args.data_root / "manifests" / "all.jsonl", rows)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
