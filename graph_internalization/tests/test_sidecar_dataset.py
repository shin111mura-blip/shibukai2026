import pytest

from graph_internalization.config import BUNDLE_INDEX
from graph_internalization.config import CONDITION_SPECS, LOCKED_MANIFESTS
from graph_internalization.sidecar_dataset import assert_same_keys_for_conditions, load_bundle_rows, manifest_selection


def test_manifest_filtering_and_overlap_zero_seed101():
    missing = [p for p in [BUNDLE_INDEX, LOCKED_MANIFESTS[101]] if not p.exists()]
    if missing:
        pytest.skip(f"external sidecar artifacts are not available: {missing}")
    rows = load_bundle_rows()
    selection = manifest_selection(LOCKED_MANIFESTS[101], rows)
    assert selection.overlap_with_graph_generator_train == 0
    assert len(selection.sample_keys) == 5889
    result = assert_same_keys_for_conditions(selection, CONDITION_SPECS)
    assert result["sample_count"] == 5889
