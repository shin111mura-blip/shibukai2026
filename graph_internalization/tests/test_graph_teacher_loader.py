import pytest

from graph_internalization.config import GRAPH_SPEC_PATH, ONTOLOGY_PATH, PRIMARY_LOCK_PATH
from graph_internalization.graph_teacher_loader import load_depth_free_graph_teacher


def test_graph_teacher_strict_load_and_freeze():
    missing = [p for p in [PRIMARY_LOCK_PATH, GRAPH_SPEC_PATH, ONTOLOGY_PATH] if not p.exists()]
    if missing:
        pytest.skip(f"external graph teacher artifacts are not available: {missing}")
    loaded = load_depth_free_graph_teacher(device="cpu")
    assert loaded.spec.class_name == "OpenVLAOnlyPooledMLP3DGraphGenerator"
    assert sum(p.numel() for p in loaded.model.parameters() if p.requires_grad) == 0
