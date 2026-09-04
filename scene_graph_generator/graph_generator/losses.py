from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


if torch is not None:
    def graph_generator_loss(node_logits, edge_logits, y_node, y_edge, validity_mask, *, node_weight=1.0, edge_weight=1.0, edge_pos_weight=None):
        node_loss = F.binary_cross_entropy_with_logits(node_logits, y_node, reduction="none").mean()
        valid = validity_mask.to(edge_logits.device).bool().unsqueeze(0)
        present_pair = (y_node.unsqueeze(2).bool() & y_node.unsqueeze(1).bool()).unsqueeze(-1)
        edge_mask = valid & present_pair
        if edge_pos_weight is not None:
            pos_weight = edge_pos_weight.to(edge_logits.device).view(1, 1, 1, -1)
            raw_edge = F.binary_cross_entropy_with_logits(edge_logits, y_edge, pos_weight=pos_weight, reduction="none")
        else:
            raw_edge = F.binary_cross_entropy_with_logits(edge_logits, y_edge, reduction="none")
        edge_loss = raw_edge[edge_mask].mean() if edge_mask.any() else raw_edge.sum() * 0.0
        return {
            "loss": node_weight * node_loss + edge_weight * edge_loss,
            "node_loss": node_loss.detach(),
            "edge_loss": edge_loss.detach(),
        }

