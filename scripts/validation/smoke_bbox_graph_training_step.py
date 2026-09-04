#!/usr/bin/env python3
"""One-batch OpenVLA BBox/Scene-Graph training smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.bbox_token_encoder import BBoxTokenConfig, BBoxTokenEncoder
from prismatic.models.scene_graph_heads import SceneGraphAuxiliaryHeads, scene_graph_losses
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.util.torch_utils import set_global_seed
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


def grad_norm(module: torch.nn.Module) -> float:
    grads = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.stack(grads).norm().item()


def lora_grad_norm(model: torch.nn.Module) -> float:
    grads = [
        p.grad.detach().float().norm()
        for name, p in model.named_parameters()
        if "lora_" in name and p.grad is not None
    ]
    if not grads:
        return 0.0
    return torch.stack(grads).norm().item()


def get_train_core(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vla-path", default="openvla/openvla-7b")
    parser.add_argument("--data-root-dir", type=Path, default=Path("/workspace/data/modified_libero_rlds"))
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=12)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lambda-edge", type=float, default=0.1)
    parser.add_argument("--lambda-between", type=float, default=0.1)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for this smoke test."
    device = torch.device("cuda:0")
    set_global_seed(42)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    vla = OpenVLAForActionPrediction.from_pretrained(
        args.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    hidden_dim = vla.config.text_config.hidden_size
    vla = get_peft_model(
        vla,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=min(args.lora_rank, 16),
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        ),
    )
    base_vla = get_train_core(vla)
    base_vla.bbox_token_encoder = BBoxTokenEncoder(
        vision_dim=hidden_dim,
        llm_dim=hidden_dim,
        config=BBoxTokenConfig(enabled=True, max_objects=args.max_objects),
    ).to(device)
    base_vla.scene_graph_heads = SceneGraphAuxiliaryHeads(hidden_dim=hidden_dim).to(device)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in args.vla_path else VicunaV15ChatPromptBuilder,
    )
    dataset = RLDSDataset(
        args.data_root_dir,
        args.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=128,
        image_aug=False,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=0)))
    required = {"bboxes_normalized", "object_mask", "edge_labels", "between_labels", "image_id"}
    missing = required - set(batch)
    if missing:
        raise KeyError(f"Missing BBox/Scene Graph batch keys: {sorted(missing)}")

    bbox_token_inputs = {
        "bboxes_normalized": batch["bboxes_normalized"].to(device),
        "object_mask": batch["object_mask"].to(device),
        "confidences": batch["confidences"].to(device),
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            labels=batch["labels"],
            bbox_token_inputs=bbox_token_inputs,
            bbox_mode="full",
            output_hidden_states=True,
            output_projector_features=True,
        )
        object_mask = batch["object_mask"].to(device).bool()
        start = output.token_layout.object_start
        stop = start + output.token_layout.num_object_tokens
        object_hidden = output.hidden_states[-1][:, start:stop]
        graph_output = base_vla.scene_graph_heads(object_hidden, object_mask)
        graph_losses = scene_graph_losses(
            graph_output,
            batch["edge_labels"].to(device),
            batch["between_labels"].to(device),
            object_mask,
            lambda_edge=args.lambda_edge,
            lambda_between=args.lambda_between,
        )
        total_loss = output.loss + graph_losses["graph_loss"]

    total_loss.backward()
    print(
        {
            "image_id": batch["image_id"][0],
            "num_objects": int(batch["object_mask"][0].sum().item()),
            "action_loss": float(output.loss.detach().float().cpu()),
            "edge_loss": float(graph_losses["edge_loss"].detach().float().cpu()),
            "between_loss": float(graph_losses["between_loss"].detach().float().cpu()),
            "total_loss": float(total_loss.detach().float().cpu()),
            "num_object_tokens": output.token_layout.num_object_tokens,
            "lora_grad_norm": lora_grad_norm(vla),
            "bbox_token_encoder_grad_norm": grad_norm(base_vla.bbox_token_encoder),
            "scene_graph_heads_grad_norm": grad_norm(base_vla.scene_graph_heads),
        }
    )


if __name__ == "__main__":
    main()
