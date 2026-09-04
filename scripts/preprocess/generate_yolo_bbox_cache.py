#!/usr/bin/env python3
"""Generate a task-level JSONL BBox cache with YOLO-World in a separate environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


def load_vocab(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def spatial_sort(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(det: Dict[str, Any]) -> tuple[float, float]:
        x1, y1, x2, y2 = det["bbox_normalized"]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    return sorted(detections, key=key)


def normalize_box(box: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    ]


def checkpoint_hash(path: str) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def resolve_image_path(path: str) -> Path:
    image_path = Path(path)
    if image_path.exists():
        return image_path
    workspace_prefix = "/workspace/"
    if path.startswith(workspace_prefix):
        fallback = Path.cwd() / path[len(workspace_prefix) :]
        if fallback.exists():
            return fallback
    return image_path


def init_yolo_world(config: str, checkpoint: str, vocab: List[str], device: str):
    try:
        from mmdet.apis import init_detector
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mmdet/mmengine are not available. Run this script inside the YOLO-World Docker environment, "
            "not inside the OpenVLA environment."
        ) from exc

    model = init_detector(config, checkpoint, device=device)
    if hasattr(model, "set_classes"):
        model.set_classes(vocab)
    elif hasattr(model, "dataset_meta"):
        model.dataset_meta["classes"] = tuple(vocab)
    return model


def run_detector(model, image_path: Path, vocab: List[str], confidence_threshold: float) -> List[Dict[str, Any]]:
    from mmdet.apis import inference_detector

    result = inference_detector(model, str(image_path))
    instances = result.pred_instances
    bboxes = instances.bboxes.detach().cpu().numpy()
    scores = instances.scores.detach().cpu().numpy()
    labels = instances.labels.detach().cpu().numpy()
    detections = []
    for bbox, score, label in zip(bboxes, scores, labels):
        if float(score) < confidence_threshold:
            continue
        detections.append(
            {
                "category": vocab[int(label)] if int(label) < len(vocab) else str(int(label)),
                "bbox_xyxy": [float(x) for x in bbox.tolist()],
                "confidence": float(score),
            }
        )
    return detections


def init_ultralytics_yolo_world(checkpoint: str, vocab: List[str]):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ultralytics is not available. Run this script inside the YOLO-World preprocessing Docker environment."
        ) from exc
    model = YOLO(checkpoint)
    model.set_classes(vocab)
    return model


def run_ultralytics_detector(
    model,
    image_path: Path,
    vocab: List[str],
    confidence_threshold: float,
    nms_threshold: float,
    device: str,
) -> List[Dict[str, Any]]:
    results = model.predict(
        source=str(image_path),
        conf=confidence_threshold,
        iou=nms_threshold,
        device=device.replace("cuda:", ""),
        verbose=False,
    )
    detections = []
    if not results:
        return detections
    boxes = results[0].boxes
    if boxes is None:
        return detections
    for xyxy, score, cls_id in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
        label = int(cls_id)
        detections.append(
            {
                "category": vocab[label] if label < len(vocab) else str(label),
                "bbox_xyxy": [float(x) for x in xyxy.tolist()],
                "confidence": float(score),
            }
        )
    return detections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("cache/yolo_bbox"))
    parser.add_argument("--detector-name", default="yolo-world")
    parser.add_argument("--backend", choices=["mmdet", "ultralytics"], default="mmdet")
    parser.add_argument("--config", default="", help="YOLO-World MMDetection config path inside the YOLO container.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--yolo-world-root", type=Path, default=Path("YOLO-World"))
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "libero_spatial_bbox_cache.jsonl"
    metadata = {
        "detector_name": args.detector_name,
        "backend": args.backend,
        "checkpoint": args.checkpoint,
        "checkpoint_hash": checkpoint_hash(args.checkpoint),
        "fixed_vocabulary": vocab,
        "confidence_threshold": args.confidence_threshold,
        "nms_threshold": args.nms_threshold,
        "code_git_commit_openvla": git_commit(Path("openvla")),
        "code_git_commit_yolo_world": git_commit(args.yolo_world_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.backend == "ultralytics":
        model = init_ultralytics_yolo_world(args.checkpoint, vocab)
    else:
        model = init_yolo_world(args.config, args.checkpoint, vocab, args.device)

    with open(args.manifest) as f, open(out_path, "w") as out:
        for line in f:
            row = json.loads(line)
            image_path = resolve_image_path(row["image_path"])
            image = Image.open(image_path)
            width, height = image.size
            if args.backend == "ultralytics":
                raw_detections = run_ultralytics_detector(
                    model,
                    image_path,
                    vocab,
                    args.confidence_threshold,
                    args.nms_threshold,
                    args.device,
                )
            else:
                raw_detections = run_detector(model, image_path, vocab, args.confidence_threshold)
            detections = []
            for det in raw_detections:
                bbox_xyxy = [float(x) for x in det["bbox_xyxy"]]
                detections.append(
                    {
                        "category": det["category"],
                        "bbox_xyxy": bbox_xyxy,
                        "bbox_normalized": normalize_box(bbox_xyxy, width, height),
                        "confidence": float(det["confidence"]),
                    }
                )
            out.write(
                json.dumps(
                    {
                        "image_id": row["image_id"],
                        "image_width": width,
                        "image_height": height,
                        "detections": spatial_sort(detections),
                        "metadata": metadata,
                    }
                )
                + "\n"
            )
    with open(args.output_dir / "bbox_cache_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
