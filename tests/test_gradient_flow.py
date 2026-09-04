import torch

from prismatic.models.scene_graph_heads import SceneGraphAuxiliaryHeads, scene_graph_losses


def test_graph_loss_flows_to_shared_lora_like_parameters():
    shared = torch.nn.Linear(4, 4, bias=False)
    heads = SceneGraphAuxiliaryHeads(hidden_dim=4)
    raw = torch.randn(1, 3, 4)
    object_hidden = shared(raw)
    object_mask = torch.tensor([[True, True, True]])
    out = heads(object_hidden, object_mask)
    edge_labels = torch.zeros_like(out.edge_logits)
    between_labels = torch.zeros_like(out.between_logits)
    losses = scene_graph_losses(out, edge_labels, between_labels, object_mask)
    losses["graph_loss"].backward()
    assert shared.weight.grad is not None
    assert torch.linalg.vector_norm(shared.weight.grad) > 0
