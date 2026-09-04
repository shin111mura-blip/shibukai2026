from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ONTOLOGY_PATH, read_json
from .graph_teacher_loader import LoadedGraphTeacher
from .losses import graph_auxiliary_loss

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    @dataclass(frozen=True)
    class ExtractedTeacherFeatures:
        token_features: torch.Tensor
        attention_mask: torch.Tensor
        token_type_mask: torch.Tensor


    def extract_teacher_features_from_openvla_output(output: Any, *, feature_layer: int = -2) -> ExtractedTeacherFeatures:
        if getattr(output, "hidden_states", None) is None:
            raise ValueError("OpenVLA output must include hidden_states for graph auxiliary training.")
        layout = getattr(output, "token_layout", None)
        if layout is None:
            raise ValueError("OpenVLA output must include token_layout for graph auxiliary training.")
        hidden = output.hidden_states[feature_layer]
        image = hidden[:, 1 : 1 + layout.num_visual_tokens, :]
        language_start = layout.language_start
        language_stop = layout.action_start if layout.action_start is not None else hidden.shape[1]
        if language_stop <= language_start:
            raise ValueError(f"No instruction tokens found: start={language_start} stop={language_stop}")
        instruction = hidden[:, language_start:language_stop, :]
        features = torch.cat([image, instruction], dim=1)
        batch = features.shape[0]
        image_types = torch.ones((batch, image.shape[1]), dtype=torch.long, device=features.device)
        instr_types = torch.full((batch, instruction.shape[1]), 2, dtype=torch.long, device=features.device)
        token_type = torch.cat([image_types, instr_types], dim=1)
        attention = torch.ones((batch, features.shape[1]), dtype=torch.bool, device=features.device)
        return ExtractedTeacherFeatures(features, attention, token_type)


    class GraphAuxiliaryModule(nn.Module):
        def __init__(self, loaded_teacher: LoadedGraphTeacher, lambda_graph: float = 0.1):
            super().__init__()
            self.teacher = loaded_teacher.model
            self.register_buffer("edge_pos_weight", loaded_teacher.edge_pos_weight.detach().clone())
            self.lambda_graph = float(lambda_graph)
            self.spec = loaded_teacher.spec
            self.ontology = read_json(ONTOLOGY_PATH)

        def forward(self, openvla_output: Any, graph_targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            extracted = extract_teacher_features_from_openvla_output(
                openvla_output,
                feature_layer=self.spec.feature_layer,
            )
            teacher_output = self.teacher(
                extracted.token_features,
                extracted.attention_mask,
                extracted.token_type_mask,
            )
            losses = graph_auxiliary_loss(
                teacher_output,
                graph_targets,
                self.ontology,
                self.edge_pos_weight,
                xyz_weight=self.spec.xyz_weight,
            )
            losses["loss_graph_weighted"] = losses["loss_graph_total"] * self.lambda_graph
            return losses
