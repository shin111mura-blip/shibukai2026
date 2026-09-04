from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import assert_locked_graph_teacher_files, load_locked_graph_teacher_spec

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class LoadedGraphTeacher:
    model: Any
    edge_pos_weight: Any
    spec: Any
    checkpoint_payload: dict[str, Any]


def _state_dict_from_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("model_state_dict", "state_dict", "model"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    if all(hasattr(v, "shape") for v in payload.values()):
        return payload
    raise KeyError("Could not locate model state dict in graph teacher checkpoint.")


def load_depth_free_graph_teacher(device: str = "cpu", freeze: bool = True) -> LoadedGraphTeacher:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required to load the graph teacher")
    assert_locked_graph_teacher_files()
    spec = load_locked_graph_teacher_spec()
    from scene_graph_generator.graph_generator.models.depth_augmented import OpenVLAOnlyPooledMLP3DGraphGenerator

    model = OpenVLAOnlyPooledMLP3DGraphGenerator(
        spec.openvla_dim,
        spec.num_nodes,
        spec.num_predicates,
        hidden_dim=spec.hidden_dim,
        num_layers=spec.num_layers,
        dropout=spec.dropout,
    )
    payload = torch.load(spec.checkpoint, map_location="cpu")
    state_dict = _state_dict_from_checkpoint(payload)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    edge_pos_weight = torch.tensor(spec.edge_pos_weight, dtype=torch.float32, device=device)
    return LoadedGraphTeacher(model=model, edge_pos_weight=edge_pos_weight, spec=spec, checkpoint_payload=payload)
