#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rollout_collection_common import DATA_ROOT, read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create simple RGB + graph-triplet sample panels.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--max-samples", type=int, default=32)
    args = parser.parse_args()
    out_dir = args.data_root / "stats" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.data_root / "manifests" / "all.jsonl")
    written = []
    for row in rows[: args.max_samples]:
        episode_dir = Path(row["episode_dir"])
        meta = read_json(episode_dir / "metadata.json")
        data = np.load(episode_dir / "frames.npz", allow_pickle=True)
        if data["rgb"].shape[0] == 0:
            continue
        idx = min(data["rgb"].shape[0] - 1, max(0, data["rgb"].shape[0] // 2))
        img = Image.fromarray(data["rgb"][idx])
        panel = Image.new("RGB", (img.width, img.height + 160), "white")
        panel.paste(img, (0, 0))
        draw = ImageDraw.Draw(panel)
        frame = meta["frames"][idx]
        triplets = meta["oracle_graph_triplets"][idx][:8]
        text = [
            f"task={meta['task_id']} policy={meta['policy_id']}",
            f"episode={meta['episode_id']} t={idx} success={meta['episode_success']} failure={meta['failure_category']}",
            f"grasping={frame['has_grasping']} contacts={frame['contact_count']}",
        ]
        text.extend(" - ".join(t) for t in triplets)
        draw.multiline_text((8, img.height + 8), "\n".join(text), fill=(0, 0, 0))
        out = out_dir / f"{meta['episode_id']}_t{idx:04d}.png"
        panel.save(out)
        written.append(str(out))
    write_json(args.data_root / "stats" / "samples" / "sample_visualization_summary.json", {"written": written})
    print(f"wrote {len(written)} samples to {out_dir}")


if __name__ == "__main__":
    main()
