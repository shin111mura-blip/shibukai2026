from scene_graph_generator.graph_generator.masks import relation_validity_mask


def test_grasping_only_gripper_to_object():
    ontology = {
        "nodes": {
            "gripper": {"index": 0, "category": "gripper", "entity_type": "gripper"},
            "obj": {"index": 1, "category": "obj", "entity_type": "object"},
            "fixture": {"index": 2, "category": "fixture", "entity_type": "fixture"},
        },
        "predicates": {"left_of": 0, "grasping": 1},
    }
    mask = relation_validity_mask(ontology)
    assert mask[0, 1, 1]
    assert not mask[1, 0, 1]
    assert not mask[2, 1, 1]
    assert not mask[0, 0, 0]

