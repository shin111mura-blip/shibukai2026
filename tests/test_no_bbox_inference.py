from pathlib import Path


def test_default_inference_uses_no_bbox():
    source = Path("openvla/experiments/robot/libero/run_libero_eval.py").read_text()
    assert 'bbox_mode: str = "none"' in source
    assert "bbox_mode=none: YOLO, BBox cache" in source
