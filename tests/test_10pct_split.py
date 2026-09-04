from scripts.preprocess.create_libero_10pct_split import select_demo_ids


def test_10pct_split_is_deterministic_and_minimum_one():
    demos = list(range(17))
    first = select_demo_ids(demos, fraction=0.10, seed=42)
    second = select_demo_ids(demos, fraction=0.10, seed=42)
    assert first == second
    assert len(first) == 2
    assert select_demo_ids([5], fraction=0.10, seed=42) == [5]
