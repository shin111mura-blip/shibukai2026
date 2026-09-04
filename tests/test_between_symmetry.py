from prismatic.vla.scene_graph.relations import RelationThresholds, is_between
from prismatic.vla.scene_graph.schema import SceneGraphNode


def node(idx, box):
    x1, y1, x2, y2 = box
    return SceneGraphNode(idx, 0, "obj", box, ((x1 + x2) / 2, (y1 + y2) / 2), x2 - x1, y2 - y1, 1.0)


def test_between_reference_order_symmetry():
    target = node(0, (0.48, 0.48, 0.52, 0.52))
    ref1 = node(1, (0.1, 0.48, 0.14, 0.52))
    ref2 = node(2, (0.86, 0.48, 0.9, 0.52))
    thresholds = RelationThresholds()
    assert is_between(target, ref1, ref2, thresholds)
    assert is_between(target, ref2, ref1, thresholds)
