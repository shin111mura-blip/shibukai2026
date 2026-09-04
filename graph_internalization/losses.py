from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


if torch is not None:
    from scene_graph_generator.graph_generator.losses_3d import graph_generator_3d_loss
    from scene_graph_generator.graph_generator.masks import relation_validity_mask


def graph_auxiliary_loss(
    teacher_output: dict[str, Any],
    graph_targets: dict[str, Any],
    ontology: dict[str, Any],
    edge_pos_weight: Any,
    *,
    xyz_weight: float = 1.0,
) -> dict[str, Any]:
    validity = torch.as_tensor(relation_validity_mask(ontology), dtype=torch.bool, device=teacher_output["edge_logits"].device)
    losses = graph_generator_3d_loss(
        teacher_output["node_logits"],
        teacher_output["edge_logits"],
        teacher_output["xyz"],
        graph_targets["y_node"].to(teacher_output["node_logits"].device),
        graph_targets["y_edge"].to(teacher_output["edge_logits"].device),
        graph_targets["y_xyz"].to(teacher_output["xyz"].device),
        graph_targets["y_xyz_mask"].to(teacher_output["xyz"].device),
        validity,
        edge_pos_weight=edge_pos_weight,
        xyz_weight=xyz_weight,
    )
    return {
        "loss_graph_total": losses["loss"],
        "loss_graph_node": losses["node_loss"],
        "loss_graph_edge": losses["edge_loss"],
        "loss_graph_relation": losses["edge_loss"],
        "loss_graph_coordinate": losses["xyz_loss"],
    }
