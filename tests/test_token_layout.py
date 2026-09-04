import torch

from prismatic.vla.token_layout import build_multimodal_with_optional_object_tokens


def test_token_layout_action_start_with_bbox_tokens():
    text = torch.randn(1, 5, 8)
    patches = torch.randn(1, 4, 8)
    objects = torch.randn(1, 2, 8)
    attention = torch.ones(1, 5, dtype=torch.bool)
    labels = torch.full((1, 5), -100)
    labels[0, 3:] = torch.tensor([10, 11])
    _, out_attention, out_labels, layout = build_multimodal_with_optional_object_tokens(
        text, attention, labels, patches, objects, torch.tensor([[True, False]])
    )
    assert layout.num_visual_tokens == 4
    assert layout.num_object_tokens == 2
    assert layout.object_start == 5
    assert layout.action_start == 1 + 4 + 2 + 2
    assert out_attention[0, layout.object_start].item() is True
    assert out_attention[0, layout.object_start + 1].item() is False
    assert torch.all(out_labels[:, layout.object_start : layout.object_start + 2] == -100)
