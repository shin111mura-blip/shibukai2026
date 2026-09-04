from scene_graph.relation_rules import SpatialThresholds, spatial_edges, structural_spatial_edges
from scene_graph.schema import Node


def node(node_id, x, y, visible=True):
    return Node(node_id, "thing", "object", True, visible, 100, (x, y))


def test_left_right_inverse():
    nodes = [node("a", 10, 50), node("b", 100, 50)]
    edges = {(e.subject, e.predicate, e.object) for e in spatial_edges(nodes, SpatialThresholds(200, 200), True)}
    assert ("a", "left_of", "b") in edges
    assert ("b", "right_of", "a") in edges


def test_above_below_inverse():
    nodes = [node("a", 50, 10), node("b", 50, 100)]
    edges = {(e.subject, e.predicate, e.object) for e in spatial_edges(nodes, SpatialThresholds(200, 200), True)}
    assert ("a", "above", "b") in edges
    assert ("b", "below", "a") in edges


def test_dead_zone_has_no_relation():
    nodes = [node("a", 50, 50), node("b", 54, 54)]
    edges = {(e.subject, e.predicate, e.object) for e in spatial_edges(nodes, SpatialThresholds(200, 200), True)}
    assert ("a", "left_of", "b") not in edges
    assert ("a", "right_of", "b") not in edges
    assert ("a", "above", "b") not in edges
    assert ("a", "below", "b") not in edges


def test_non_visible_nodes_excluded_from_observable_spatial_edges():
    nodes = [node("a", 10, 50, visible=True), node("b", 100, 50, visible=False)]
    assert spatial_edges(nodes, SpatialThresholds(200, 200), visible_only=True) == []


def test_structural_edges_use_supplied_image_plane_axes():
    nodes = [node("bowl_2", 0, 0), node("ramekin", 0, 0), node("stove", 0, 0)]
    positions = {
        "bowl_2": (33.5, 120.1),
        "ramekin": (69.4, 121.0),
        "stove": (160.7, 148.1),
    }
    edges = {(edge.subject, edge.predicate, edge.object) for edge in structural_spatial_edges(nodes, positions)}
    assert ("bowl_2", "left_of", "ramekin") in edges
    assert ("ramekin", "right_of", "bowl_2") in edges
    assert ("ramekin", "left_of", "stove") in edges
    assert ("stove", "right_of", "ramekin") in edges
    assert ("bowl_2", "above", "ramekin") not in edges


def test_structural_edges_include_nearest_frontback_from_camera_depth():
    nodes = [node("front", 50, 50), node("middle", 55, 80), node("back", 60, 110)]
    positions = {"front": (50.0, 50.0), "middle": (55.0, 80.0), "back": (60.0, 110.0)}
    depths = {"front": 1.0, "middle": 1.2, "back": 1.5}
    edges = {(edge.subject, edge.predicate, edge.object) for edge in structural_spatial_edges(nodes, positions, depth_positions=depths)}
    assert ("front", "front_of", "middle") in edges
    assert ("middle", "behind", "front") in edges
    assert ("middle", "front_of", "back") in edges
    assert ("back", "behind", "middle") in edges
