import numpy as np

from scripts.rollout_collection_common import build_schema_lock


def test_schema_lock_shape_matches_ontology_counts():
    schema = build_schema_lock()
    assert len(schema["GRAPH_TARGET_SHAPE"]) == 3
    assert schema["GRAPH_TARGET_SHAPE"] == schema["RELATION_MASK_SHAPE"]
    assert schema["NODE_MASK_SHAPE"][0] == schema["GRAPH_TARGET_SHAPE"][0]
    assert "touching" not in schema["relation_order"]
