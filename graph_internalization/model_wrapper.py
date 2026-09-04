from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import CONDITION_SPECS
from .depth_adapter import RGBDToRGBAdapter
from .graph_auxiliary import GraphAuxiliaryModule
from .graph_teacher_loader import load_depth_free_graph_teacher

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class WrapperConfig:
    condition: str
    lambda_graph: float = 0.1
    depth_adapter_lr: float = 5e-4

    @property
    def uses_depth(self) -> bool:
        return CONDITION_SPECS[self.condition]["uses_depth"]

    @property
    def uses_graph_aux(self) -> bool:
        return CONDITION_SPECS[self.condition]["uses_graph_aux"]

    @property
    def uses_action_loss(self) -> bool:
        return CONDITION_SPECS[self.condition].get("uses_action_loss", True)


if nn is not None:

    class OpenVLAGraphInternalizationWrapper(nn.Module):
        """Training wrapper that keeps graph predictions out of the action decoder."""

        def __init__(self, vla: nn.Module, cfg: WrapperConfig, device: str = "cpu"):
            super().__init__()
            if cfg.condition not in CONDITION_SPECS:
                raise ValueError(f"Unknown condition {cfg.condition}")
            self.vla = vla
            self.cfg = cfg
            self.depth_adapter = RGBDToRGBAdapter() if cfg.uses_depth else None
            self.graph_aux = None
            if cfg.uses_graph_aux:
                self.graph_aux = GraphAuxiliaryModule(load_depth_free_graph_teacher(device=device), cfg.lambda_graph)

        def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
            pixel_values = batch["pixel_values"]
            if self.depth_adapter is not None:
                pixel_values = self.depth_adapter(pixel_values, batch.get("depth"))
            elif "depth" in batch and self.cfg.condition.startswith("rgb_"):
                pass
            forward_kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "pixel_values": pixel_values,
                "labels": batch.get("labels"),
            }
            if self.cfg.uses_graph_aux:
                forward_kwargs["output_hidden_states"] = True
                forward_kwargs["output_projector_features"] = True
                forward_kwargs["return_dict"] = True
            output = self.vla(**forward_kwargs)
            if self.cfg.uses_action_loss:
                action_loss = output.loss
                total_loss = action_loss
            else:
                action_loss = output.loss.detach()
                total_loss = None
            result = {"openvla_output": output, "loss_action": action_loss}
            if self.cfg.uses_graph_aux:
                if "graph_targets" not in batch:
                    raise ValueError("graph_targets are required for graph auxiliary training.")
                graph_losses = self.graph_aux(output, batch["graph_targets"])
                result.update(graph_losses)
                result["loss_total"] = (
                    total_loss + graph_losses["loss_graph_weighted"]
                    if total_loss is not None
                    else graph_losses["loss_graph_weighted"]
                )
            elif total_loss is not None:
                result["loss_total"] = total_loss
            else:
                raise ValueError("At least one of action loss or graph auxiliary loss must be enabled.")
            return result

        def inference_kwargs(self, batch: dict[str, Any]) -> dict[str, Any]:
            if self.cfg.uses_depth and "depth" not in batch:
                raise ValueError(f"{self.cfg.condition} requires depth at inference.")
            pixel_values = batch["pixel_values"]
            if self.depth_adapter is not None:
                pixel_values = self.depth_adapter(pixel_values, batch["depth"])
            return {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "pixel_values": pixel_values,
            }
