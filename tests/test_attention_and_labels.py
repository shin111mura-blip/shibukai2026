import torch

from prismatic.vla.token_layout import build_multimodal_with_optional_object_tokens


def test_bbox_token_labels_are_ignored_and_padding_not_attended():
    embeds = torch.zeros(1, 4, 3)
    patches = torch.zeros(1, 2, 3)
    objects = torch.zeros(1, 3, 3)
    attention = torch.tensor([[True, True, True, False]])
    labels = torch.tensor([[-100, -100, 7, 8]])
    _, mask, out_labels, layout = build_multimodal_with_optional_object_tokens(
        embeds, attention, labels, patches, objects, torch.tensor([[True, False, True]])
    )
    assert mask.tolist() == [[True, True, True, True, False, True, True, True, False]]
    assert out_labels[0, layout.object_start : layout.object_start + 3].tolist() == [-100, -100, -100]
