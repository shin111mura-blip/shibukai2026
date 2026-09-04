from scene_graph_generator.scripts.create_splits import split_counts


def test_split_counts_keep_all_splits_for_reasonable_task_size():
    assert split_counts(4) == (2, 1, 1)
    assert sum(split_counts(43)) == 43
    assert all(x > 0 for x in split_counts(43))

