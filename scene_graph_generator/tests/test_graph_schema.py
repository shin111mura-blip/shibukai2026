from scene_graph_generator.graph_generator.schema import canonical_graph_key, validate_graph


def test_graph_schema_allows_sparse_recorded_edges_only():
    graph = {
        "nodes": [
            {"id": "gripper_1", "category": "gripper", "entity_type": "gripper", "present": True},
            {"id": "bowl_1", "category": "bowl", "entity_type": "object", "present": True},
        ],
        "binary_edges": [{"subject": "gripper_1", "predicate": "grasping", "object": "bowl_1"}],
    }
    assert validate_graph(graph) == []
    assert canonical_graph_key(graph) == canonical_graph_key({"nodes": list(reversed(graph["nodes"])), "binary_edges": graph["binary_edges"]})

