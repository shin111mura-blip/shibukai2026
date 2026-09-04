from scripts.rollout_collection_common import WORLD_GRAPH_ROOT, TEACHER_GRAPH_3D_ROOT


def test_existing_oracle_roots_are_present():
    assert WORLD_GRAPH_ROOT.exists()
    assert TEACHER_GRAPH_3D_ROOT.exists()
