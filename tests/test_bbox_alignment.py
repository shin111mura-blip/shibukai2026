import torch

from prismatic.vla.bbox_alignment import ImageTransformGeometry, bboxes_to_patch_mask, transform_bboxes_to_model_input


def test_letterbox_bbox_alignment_and_patch_mask():
    geom = ImageTransformGeometry(original_width=100, original_height=50, input_width=224, input_height=224)
    box = torch.tensor([[0.0, 0.0, 100.0, 50.0]])
    aligned = transform_bboxes_to_model_input(box, geom)
    assert torch.allclose(aligned, torch.tensor([[0.0, 0.25, 1.0, 0.75]]))
    mask = bboxes_to_patch_mask(aligned, (4, 4))
    assert mask.shape == (1, 16)
    assert mask.any()
