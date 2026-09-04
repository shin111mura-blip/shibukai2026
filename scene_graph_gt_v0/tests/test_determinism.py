from scene_graph.canonicalize import make_graph, sha256_payload
from scene_graph.schema import Edge, Node


def test_deterministic_hash_ignores_input_order():
    nodes_a = [Node("b", "thing", "object", True, True), Node("a", "thing", "object", True, True)]
    nodes_b = list(reversed(nodes_a))
    edges_a = [Edge("b", "right_of", "a"), Edge("a", "left_of", "b")]
    edges_b = list(reversed(edges_a))
    graph_a = make_graph(source="rule_based", mode="world", task_id="0", demo_id="demo_0", frame_id=1, nodes=nodes_a, edges=edges_a, config_hash="cfg")
    graph_b = make_graph(source="rule_based", mode="world", task_id="0", demo_id="demo_0", frame_id=1, nodes=nodes_b, edges=edges_b, config_hash="cfg")
    assert sha256_payload(graph_a) == sha256_payload(graph_b)

