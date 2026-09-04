#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .common import DEFAULT_OUTPUT_ROOT
except ImportError:  # pragma: no cover
    from common import DEFAULT_OUTPUT_ROOT
from scene_graph_generator.graph_generator.schema import write_json


def load_history(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def series(history: list[dict], key: str, split: str) -> list[float]:
    out = []
    for row in history:
        cur = row[split]
        for part in key.split("."):
            cur = cur[part]
        out.append(float(cur))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--architecture", default="pooled_mlp_depth_3d")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_dir = args.output_root / "metrics" / args.architecture
    history = load_history(metric_dir / "training_history.json")
    summary_path = metric_dir / "training_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    epochs = [int(row["epoch"]) for row in history]
    best_epoch = summary.get("best_epoch")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=160)
    axes = axes.reshape(-1)

    axes[0].plot(epochs, series(history, "loss", "train"), label="train total")
    axes[0].plot(epochs, series(history, "loss", "validation"), label="validation total")
    axes[0].plot(epochs, series(history, "edge_loss", "validation"), label="validation edge")
    axes[0].plot(epochs, series(history, "xyz_loss", "validation"), label="validation xyz")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=8)

    axes[1].plot(epochs, series(history, "metrics.triplet.f1", "validation"), label="triplet micro F1")
    axes[1].plot(epochs, series(history, "metrics.triplet.macro_f1", "validation"), label="triplet macro F1")
    axes[1].plot(epochs, series(history, "metrics.graph.jaccard_similarity", "validation"), label="graph Jaccard")
    axes[1].set_title("Relation Metrics")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(fontsize=8)

    axes[2].plot(epochs, series(history, "xyz_metrics.rmse", "validation"), label="RMSE")
    axes[2].plot(epochs, series(history, "xyz_metrics.mae", "validation"), label="MAE")
    axes[2].plot(epochs, series(history, "xyz_metrics.mean_l2", "validation"), label="mean L2")
    axes[2].set_title("3D Coordinate Error")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("meter")
    axes[2].legend(fontsize=8)

    axes[3].plot(epochs, series(history, "xyz_metrics.within_5cm", "validation"), label="within 5cm")
    axes[3].plot(epochs, series(history, "xyz_metrics.within_10cm", "validation"), label="within 10cm")
    axes[3].plot(epochs, series(history, "metrics.graph.exact_match", "validation"), label="graph exact")
    axes[3].set_title("Accuracy-Like Metrics")
    axes[3].set_xlabel("epoch")
    axes[3].set_ylim(0.0, 1.02)
    axes[3].legend(fontsize=8)

    if best_epoch is not None:
        for ax in axes:
            ax.axvline(int(best_epoch), color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.grid(True, alpha=0.25)

    fig.suptitle(f"{args.architecture} training curves; best epoch={best_epoch}")
    fig.tight_layout()

    out_dir = args.output_root / "visualizations" / args.architecture / "training_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loss_and_metrics.png"
    fig.savefig(out_path)
    plt.close(fig)

    compact = {
        "status": "ok",
        "architecture": args.architecture,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "final_train_loss": series(history, "loss", "train")[-1] if history else None,
        "best_validation_loss": min(series(history, "loss", "validation")) if history else None,
        "output": str(out_path),
    }
    write_json(out_dir / "training_curve_summary.json", compact)
    print(json.dumps({"status": "ok", "output": str(out_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
