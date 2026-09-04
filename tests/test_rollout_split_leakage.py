from collections import defaultdict


def test_group_key_prevents_initial_state_split_leakage():
    rows = [
        {"suite_name": "libero_spatial", "task_id": 0, "initial_state_id": 1, "policy_id": "a"},
        {"suite_name": "libero_spatial", "task_id": 0, "initial_state_id": 1, "policy_id": "b"},
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["suite_name"], row["task_id"], row["initial_state_id"])].append(row)
    assert len(groups) == 1
