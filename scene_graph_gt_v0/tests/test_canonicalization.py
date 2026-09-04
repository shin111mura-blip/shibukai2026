import pytest

from scene_graph.canonicalize import make_graph, sha256_payload
from scene_graph.schema import Edge, Node


def nodes():
    return [
        Node("b", "thing", "object", True, True, 1, (2.0, 2.0)),
        Node("a", "thing", "object", True, True, 1, (1.0, 1.0)),
        Node("gripper", "gripper", "gripper", True, True, 1, (0.0, 0.0)),
    ]


def test_node_and_edge_order_are_canonical_and_duplicate_edges_removed():
    graph = make_graph(
        source="rule_based",
        mode="observable",
        task_id="0",
        demo_id="demo_0",
        frame_id=0,
        nodes=nodes(),
        edges=[Edge("b", "right_of", "a"), Edge("b", "right_of", "a"), Edge("a", "left_of", "b")],
        config_hash="abc",
    )
    assert [n["id"] for n in graph["nodes"]] == ["a", "b", "gripper"]
    assert graph["binary_edges"] == [
        {"object": "b", "predicate": "left_of", "subject": "a"},
        {"object": "a", "predicate": "right_of", "subject": "b"},
    ]


def test_predicate_enum_rejects_unknown_values():
    with pytest.raises(ValueError):
        make_graph(
            source="rule_based",
            mode="observable",
            task_id="0",
            demo_id="demo_0",
            frame_id=0,
            nodes=nodes(),
            edges=[Edge("a", "touching", "b")],
            config_hash="abc",
        )


def test_frontback_predicates_are_allowed():
    graph = make_graph(
        source="rule_based",
        mode="observable",
        task_id="0",
        demo_id="demo_0",
        frame_id=0,
        nodes=nodes(),
        edges=[Edge("a", "front_of", "b"), Edge("b", "behind", "a")],
        config_hash="abc",
    )
    assert graph["binary_edges"] == [
        {"object": "b", "predicate": "front_of", "subject": "a"},
        {"object": "a", "predicate": "behind", "subject": "b"},
    ]


def test_same_input_same_hash():
    kwargs = dict(source="rule_based", mode="observable", task_id="0", demo_id="demo_0", frame_id=0, nodes=nodes(), edges=[Edge("a", "left_of", "b")], config_hash="abc")
    assert sha256_payload(make_graph(**kwargs)) == sha256_payload(make_graph(**kwargs))
