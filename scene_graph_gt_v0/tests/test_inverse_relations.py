from scene_graph.relation_rules import SpatialThresholds, spatial_edges
from scene_graph.schema import Node


def test_inverse_relations_are_generated_consistently():
    nodes = [
        Node("left", "thing", "object", True, True, 10, (10.0, 50.0)),
        Node("right", "thing", "object", True, True, 10, (90.0, 50.0)),
    ]
    edges = {(e.subject, e.predicate, e.object) for e in spatial_edges(nodes, SpatialThresholds(100, 100), True)}
    assert ("left", "left_of", "right") in edges
    assert ("right", "right_of", "left") in edges


def test_vertical_inverse_relations_are_generated_consistently():
    nodes = [
        Node("top", "thing", "object", True, True, 10, (50.0, 10.0)),
        Node("bottom", "thing", "object", True, True, 10, (50.0, 90.0)),
    ]
    edges = {(e.subject, e.predicate, e.object) for e in spatial_edges(nodes, SpatialThresholds(100, 100), True)}
    assert ("top", "above", "bottom") in edges
    assert ("bottom", "below", "top") in edges
