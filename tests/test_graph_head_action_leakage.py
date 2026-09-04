import inspect

from prismatic.models.scene_graph_heads import SceneGraphAuxiliaryHeads


def test_graph_head_forward_has_no_action_hidden_argument():
    sig = inspect.signature(SceneGraphAuxiliaryHeads.forward)
    assert "action_hidden" not in sig.parameters
    assert list(sig.parameters) == ["self", "object_hidden", "object_mask"]
