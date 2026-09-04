import pytest

from graph_internalization.config import (
    CONDITION_SPECS,
    GRAPH_SPEC_PATH,
    GRAPH_TEACHER_CLASS,
    GRAPH_TEACHER_SHA256,
    ONTOLOGY_PATH,
    PRIMARY_LOCK_PATH,
    RELATION_VOCAB_SHA256,
    assert_locked_graph_teacher_files,
)


def test_locked_graph_teacher_spec_files():
    missing = [p for p in [PRIMARY_LOCK_PATH, GRAPH_SPEC_PATH, ONTOLOGY_PATH] if not p.exists()]
    if missing:
        pytest.skip(f"external graph teacher artifacts are not available: {missing}")
    report = assert_locked_graph_teacher_files()
    assert report["primary_teacher"] == "depth_free"
    assert report["class_name"] == GRAPH_TEACHER_CLASS
    assert report["checkpoint_sha256"] == GRAPH_TEACHER_SHA256
    assert report["relation_vocabulary_sha256"] == RELATION_VOCAB_SHA256


def test_condition_contract():
    assert CONDITION_SPECS == {
        "rgb_action": {"uses_depth": False, "uses_graph_aux": False, "uses_action_loss": True},
        "rgbd_action": {"uses_depth": True, "uses_graph_aux": False, "uses_action_loss": True},
        "rgb_graph": {"uses_depth": False, "uses_graph_aux": True, "uses_action_loss": True},
        "rgb_graph_no_action": {"uses_depth": False, "uses_graph_aux": True, "uses_action_loss": False},
        "rgbd_graph": {"uses_depth": True, "uses_graph_aux": True, "uses_action_loss": True},
    }
