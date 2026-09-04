import numpy as np

from scene_graph_generator.graph_generator.decoding import decode_graph


def test_decoding_applies_node_and_validity_masks():
    ontology = {
        "nodes": {
            "gripper": {"index": 0, "category": "gripper", "entity_type": "gripper"},
            "obj": {"index": 1, "category": "obj", "entity_type": "object"},
        },
        "predicates": {"grasping": 0},
    }
    node_logits = np.array([5.0, 5.0])
    edge_logits = np.zeros((2, 2, 1))
    edge_logits[0, 1, 0] = 5.0
    edge_logits[1, 0, 0] = 5.0
    mask = np.zeros((2, 2, 1), dtype=bool)
    mask[0, 1, 0] = True
    graph = decode_graph(node_logits, edge_logits, ontology, mask)
    assert graph["binary_edges"] == [{"subject": "gripper", "predicate": "grasping", "object": "obj"}]

