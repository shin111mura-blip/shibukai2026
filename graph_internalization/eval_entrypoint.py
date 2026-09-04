from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "run_id",
    "condition",
    "seed",
    "eval_tier",
    "task_id",
    "success_rate",
    "rsa",
    "fca",
    "relation_probe_accuracy",
    "graph_validation_f1",
    "xyz_rmse",
    "inference_latency_ms",
    "additional_parameters",
    "failure_mode",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-run eval JSON/CSV files into the locked schema.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input.glob("**/eval_metrics.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    print(f"wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
