#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from rollout_collection_common import JOBS_FULL, load_config, main_arg_parser, make_jobs, write_jsonl


def main() -> None:
    parser = main_arg_parser("Build deterministic OpenVLA rollout job manifest.")
    parser.add_argument("--output", type=Path, default=JOBS_FULL)
    args = parser.parse_args()
    config = load_config(args.config)
    jobs = make_jobs(config)
    write_jsonl(args.output, jobs)
    print(f"wrote {len(jobs)} jobs to {args.output}")


if __name__ == "__main__":
    main()
