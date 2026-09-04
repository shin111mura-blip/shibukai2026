#!/usr/bin/env python3
"""Visualize YOLO BBox cache entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def draw(image: Image.Image, detections: list[dict]) -> Image.Image:
    out = image.convert("RGB").copy()
    canvas = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    w, h = out.size
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox_normalized"]
        box = (x1 * w, y1 * h, x2 * w, y2 * h)
        canvas.rectangle(box, outline=(255, 80, 0), width=2)
        canvas.text((box[0] + 2, box[1] + 2), f"{idx}:{det['category']} {det['confidence']:.2f}", fill=(255, 255, 255))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/yolo_bbox_visualization"))
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = {}
    with open(args.manifest) as f:
        for line in f:
            row = json.loads(line)
            images[row["image_id"]] = row["image_path"]

    count = 0
    with open(args.bbox_cache) as f:
        for line in f:
            row = json.loads(line)
            if row["image_id"] not in images:
                continue
            out = draw(Image.open(images[row["image_id"]]), row.get("detections", []))
            out.save(args.output_dir / f"{row['image_id'].replace('/', '__')}.png")
            count += 1
            if count >= args.max_images:
                break
    print(f"Wrote {count} overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
