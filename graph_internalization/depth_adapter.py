from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class RGBDToRGBAdapter(nn.Module):
        """1x1 RGB-D adapter initialized to preserve RGB and ignore depth."""

        def __init__(self, depth_init: float = 0.0):
            super().__init__()
            self.proj = nn.Conv2d(4, 3, kernel_size=1, bias=True)
            self.reset_parameters(depth_init=depth_init)

        def reset_parameters(self, depth_init: float = 0.0) -> None:
            with torch.no_grad():
                self.proj.weight.zero_()
                for channel in range(3):
                    self.proj.weight[channel, channel, 0, 0] = 1.0
                self.proj.weight[:, 3, 0, 0] = float(depth_init)
                self.proj.bias.zero_()

        def forward(self, pixel_values: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
            if depth is None:
                raise ValueError("Depth is required for RGB-D conditions.")
            if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
                raise ValueError(f"Expected RGB tensor [B,3,H,W], got {tuple(pixel_values.shape)}")
            if depth.ndim != 4 or depth.shape[1] != 1:
                raise ValueError(f"Expected depth tensor [B,1,H,W], got {tuple(depth.shape)}")
            if pixel_values.shape[0] != depth.shape[0] or pixel_values.shape[2:] != depth.shape[2:]:
                raise ValueError(
                    f"RGB and depth must be synchronized, got RGB={tuple(pixel_values.shape)} depth={tuple(depth.shape)}"
                )
            return self.proj(torch.cat([pixel_values, depth.to(pixel_values.dtype)], dim=1))


def depth_adapter_parameter_group(adapter: "RGBDToRGBAdapter", lr: float) -> dict:
    return {"params": [p for p in adapter.parameters() if p.requires_grad], "lr": lr, "name": "depth_adapter"}
