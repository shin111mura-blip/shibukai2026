from prismatic.vla.scene_graph import RelationThresholds, build_scene_graph


def test_scene_graph_pairwise_relations():
    detections = [
        {"category": "a", "bbox_normalized": [0.1, 0.4, 0.2, 0.5], "confidence": 0.9},
        {"category": "b", "bbox_normalized": [0.7, 0.4, 0.8, 0.5], "confidence": 0.8},
    ]
    sg = build_scene_graph("img", detections, RelationThresholds())
    rels = {(edge.source, edge.target): edge.relations for edge in sg.pairwise_edges}
    assert "left_of" in rels[(0, 1)]
    assert "right_of" in rels[(1, 0)]
