import pytest
import torch

from graph_internalization.depth_adapter import RGBDToRGBAdapter


def test_depth_adapter_identity_like_init():
    adapter = RGBDToRGBAdapter()
    rgb = torch.randn(2, 3, 16, 16)
    depth = torch.randn(2, 1, 16, 16)
    out = adapter(rgb, depth)
    assert torch.equal(out, rgb)


def test_depth_adapter_requires_depth():
    adapter = RGBDToRGBAdapter()
    rgb = torch.randn(2, 3, 16, 16)
    with pytest.raises(ValueError, match="Depth is required"):
        adapter(rgb, None)
