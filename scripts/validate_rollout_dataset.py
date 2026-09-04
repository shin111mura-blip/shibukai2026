#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rollout_collection_common import DATA_ROOT, SCHEMA_LOCK_JSON, read_json, validate_episode_dir, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate collected rollout dataset.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--schema-lock", type=Path, default=SCHEMA_LOCK_JSON)
    args = parser.parse_args()
    schema = read_json(args.schema_lock) if args.schema_lock.exists() else None
    rows = []
    quarantine = []
    for episode_dir in sorted((args.data_root / "episodes").glob("*/*/*")):
        if not episode_dir.is_dir():
            continue
        ok, errors, meta = validate_episode_dir(episode_dir, schema)
        row = {"episode_dir": str(episode_dir), "ok": ok, "errors": errors, **{k: meta.get(k) for k in ("job_id", "policy_id", "task_id", "initial_state_id", "episode_success")}}
        rows.append(row)
        if not ok:
            quarantine.append(row)
    write_json(args.data_root / "reports" / "dataset_validation.json", {"num_episodes": len(rows), "num_valid": sum(r["ok"] for r in rows), "num_invalid": len(quarantine)})
    write_jsonl(args.data_root / "manifests" / "all.jsonl", [r for r in rows if r["ok"]])
    write_jsonl(args.data_root / "manifests" / "quarantined.jsonl", quarantine)
    print(f"valid={sum(r['ok'] for r in rows)} invalid={len(quarantine)}")
    raise SystemExit(0 if not quarantine else 1)


if __name__ == "__main__":
    main()
