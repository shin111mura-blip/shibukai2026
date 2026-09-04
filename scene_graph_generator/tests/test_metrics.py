from scene_graph_generator.graph_generator.metrics import summarize_examples


def test_metrics_use_set_triplets():
    metrics = summarize_examples(
        [
            {
                "pred_nodes": {"a", "b"},
                "gt_nodes": {"a", "b", "c"},
                "pred_edges": {("a", "left_of", "b")},
                "gt_edges": {("a", "left_of", "b"), ("b", "right_of", "c")},
            }
        ]
    )
    assert metrics["node"]["tp"] == 2
    assert metrics["node"]["fn"] == 1
    assert metrics["triplet"]["tp"] == 1
    assert metrics["triplet"]["fn"] == 1

