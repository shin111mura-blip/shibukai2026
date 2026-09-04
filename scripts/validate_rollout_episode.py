#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rollout_collection_common import SCHEMA_LOCK_JSON, read_json, validate_episode_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one collected rollout episode directory.")
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--schema-lock", type=Path, default=SCHEMA_LOCK_JSON)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    schema = read_json(args.schema_lock) if args.schema_lock.exists() else None
    ok, errors, meta = validate_episode_dir(args.episode_dir, schema)
    report = {"episode_dir": str(args.episode_dir), "ok": ok, "errors": errors, "metadata": meta}
    if args.report:
        write_json(args.report, report)
    print("PASS" if ok else "FAIL")
    if errors:
        for err in errors:
            print(err)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
