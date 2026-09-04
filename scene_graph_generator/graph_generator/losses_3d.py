from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


if torch is not None:
    from .losses import graph_generator_loss

    def graph_generator_3d_loss(
        node_logits,
        edge_logits,
        xyz,
        y_node,
        y_edge,
        y_xyz,
        y_xyz_mask,
        validity_mask,
        *,
        edge_pos_weight=None,
        xyz_weight=1.0,
    ):
        losses = graph_generator_loss(
            node_logits,
            edge_logits,
            y_node,
            y_edge,
            validity_mask,
            edge_pos_weight=edge_pos_weight,
        )
        mask = y_xyz_mask.bool().unsqueeze(-1).expand_as(y_xyz)
        if mask.any():
            xyz_loss = F.smooth_l1_loss(xyz[mask], y_xyz[mask])
        else:
            xyz_loss = xyz.sum() * 0.0
        losses["xyz_loss"] = xyz_loss.detach()
        losses["loss"] = losses["loss"] + xyz_weight * xyz_loss
        return losses
