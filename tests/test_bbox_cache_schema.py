from scripts.preprocess.generate_yolo_bbox_cache import normalize_box, spatial_sort


def test_bbox_cache_schema_and_spatial_sort():
    detections = [
        {"category": "b", "bbox_normalized": [0.7, 0.1, 0.8, 0.2], "confidence": 0.9},
        {"category": "a", "bbox_normalized": [0.1, 0.5, 0.2, 0.6], "confidence": 0.1},
    ]
    ordered = spatial_sort(detections)
    assert [det["category"] for det in ordered] == ["a", "b"]
    assert normalize_box([10, 20, 30, 40], 100, 200) == [0.1, 0.1, 0.3, 0.2]
