from scene_graph_generator.graph_generator.targets import encode_targets


def test_target_encoding_multi_label_edge_tensor():
    ontology = {
        "nodes": {
            "a": {"index": 0, "category": "a", "entity_type": "object"},
            "b": {"index": 1, "category": "b", "entity_type": "object"},
        },
        "predicates": {"left_of": 0, "right_of": 1},
    }
    graph = {
        "nodes": [
            {"id": "a", "category": "a", "entity_type": "object", "present": True},
            {"id": "b", "category": "b", "entity_type": "object", "present": True},
        ],
        "binary_edges": [
            {"subject": "a", "predicate": "left_of", "object": "b"},
            {"subject": "a", "predicate": "right_of", "object": "b"},
        ],
    }
    y_node, y_edge = encode_targets(graph, ontology)
    assert y_node.tolist() == [1.0, 1.0]
    assert y_edge[0, 1, 0] == 1.0
    assert y_edge[0, 1, 1] == 1.0
    assert y_edge[1, 0, 0] == 0.0

