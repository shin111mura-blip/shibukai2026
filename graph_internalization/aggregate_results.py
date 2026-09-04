from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _float(value: str) -> float:
    return float(value) if value not in {"", None} else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paired differences and factorial effects for 4-condition runs.")
    parser.add_argument("--per-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.per_run, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_seed = defaultdict(dict)
    for row in rows:
        key = (row.get("seed"), row.get("eval_tier", ""))
        by_seed[key][row["condition"]] = row
    diff_rows = []
    for (seed, tier), conds in sorted(by_seed.items()):
        def val(condition: str) -> float | None:
            return _float(conds[condition]["success_rate"]) if condition in conds else None
        pairs = [
            ("rgbd_action-rgb_action", "rgbd_action", "rgb_action"),
            ("rgb_graph-rgb_action", "rgb_graph", "rgb_action"),
            ("rgbd_graph-rgbd_action", "rgbd_graph", "rgbd_action"),
            ("rgbd_graph-rgb_graph", "rgbd_graph", "rgb_graph"),
        ]
        values = {}
        for name, a, b in pairs:
            if val(a) is not None and val(b) is not None:
                values[name] = val(a) - val(b)
                diff_rows.append({"seed": seed, "eval_tier": tier, "contrast": name, "success_rate_delta": values[name]})
        if "rgbd_graph-rgbd_action" in values and "rgb_graph-rgb_action" in values:
            diff_rows.append(
                {
                    "seed": seed,
                    "eval_tier": tier,
                    "contrast": "interaction",
                    "success_rate_delta": values["rgbd_graph-rgbd_action"] - values["rgb_graph-rgb_action"],
                }
            )
    out = args.output_dir / "paired_differences.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "eval_tier", "contrast", "success_rate_delta"])
        writer.writeheader()
        writer.writerows(diff_rows)
    (args.output_dir / "summary.md").write_text(
        f"# Summary\n\nInput: `{args.per_run}`\n\nPaired differences: `{out}`\n\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
