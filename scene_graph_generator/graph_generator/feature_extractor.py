from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, Sequence

from .token_selection import infer_prismatic_token_layout


class OpenVLAFeatureExtractor:
    def __init__(self, checkpoint: Path, device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        torch_dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            str(checkpoint),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.checkpoint = Path(checkpoint)
        self.torch_dtype = torch_dtype

    def environment_report(self) -> Dict[str, Any]:
        import transformers
        torch = self.torch
        return {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "checkpoint": str(self.checkpoint),
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "tokenizer_class": type(getattr(self.processor, "tokenizer", None)).__name__,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }

    def extract(self, image, instruction: str, feature_layer: int = -2) -> Dict[str, Any]:
        torch = self.torch
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, image, return_tensors="pt")
        inputs = {
            k: v.to(self.device, dtype=self.torch_dtype) if torch.is_floating_point(v) else v.to(self.device)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            out = self.model(**inputs, output_hidden_states=True, output_projector_features=True, return_dict=True)
        projector = out.projector_features
        hidden = out.hidden_states[feature_layer]
        layout = infer_prismatic_token_layout(
            inputs["input_ids"],
            inputs["attention_mask"],
            int(projector.shape[1]),
            getattr(self.processor, "tokenizer", None),
        )
        image_feat = hidden[:, layout.image_start : layout.image_end, :]
        instr_feat = hidden[:, layout.instruction_positions, :]
        features = torch.cat([image_feat, instr_feat], dim=1).squeeze(0).detach().cpu()
        token_type = torch.cat(
            [
                torch.ones(image_feat.shape[1], dtype=torch.int64),
                torch.full((instr_feat.shape[1],), 2, dtype=torch.int64),
            ],
            dim=0,
        )
        attn = torch.ones(features.shape[0], dtype=torch.bool)
        tokenizer = getattr(self.processor, "tokenizer", None)
        token_strings = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].detach().cpu().tolist()) if tokenizer else []
        return {
            "features": features,
            "attention_mask": attn,
            "token_type_mask": token_type,
            "inspection": {
                "prompt": prompt,
                "input_ids": inputs["input_ids"][0].detach().cpu().tolist(),
                "tokens": token_strings,
                "input_ids_shape": list(inputs["input_ids"].shape),
                "attention_mask_shape": list(inputs["attention_mask"].shape),
                "pixel_values_shape": list(inputs["pixel_values"].shape),
                "hidden_state_count": len(out.hidden_states),
                "hidden_state_shapes": [list(x.shape) for x in out.hidden_states],
                "projector_features_shape": list(projector.shape),
                "feature_layer": feature_layer,
                "image_token_range": [layout.image_start, layout.image_end],
                "instruction_positions": layout.instruction_positions,
                "bos_positions": layout.bos_positions,
                "padding_positions": layout.padding_positions,
                "extracted_feature_shape": list(features.shape),
                "feature_mask_shape": list(attn.shape),
            },
        }

    def extract_batch(self, images: Sequence[Any], instructions: Sequence[str], feature_layer: int = -2):
        torch = self.torch
        prompts = [f"In: What action should the robot take to {instruction.lower()}?\nOut:" for instruction in instructions]
        inputs = self.processor(prompts, list(images), return_tensors="pt", padding=True)
        inputs = {
            k: v.to(self.device, dtype=self.torch_dtype) if torch.is_floating_point(v) else v.to(self.device)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            out = self.model(**inputs, output_hidden_states=True, output_projector_features=True, return_dict=True)
        projector = out.projector_features
        hidden = out.hidden_states[feature_layer]
        batch_features = []
        batch_attn = []
        batch_token_type = []
        tokenizer = getattr(self.processor, "tokenizer", None)
        for batch_index in range(hidden.shape[0]):
            layout = infer_prismatic_token_layout(
                inputs["input_ids"][batch_index : batch_index + 1],
                inputs["attention_mask"][batch_index : batch_index + 1],
                int(projector.shape[1]),
                tokenizer,
            )
            image_feat = hidden[batch_index, layout.image_start : layout.image_end, :]
            instr_feat = hidden[batch_index, layout.instruction_positions, :]
            features = torch.cat([image_feat, instr_feat], dim=0)
            token_type = torch.cat(
                [
                    torch.ones(image_feat.shape[0], dtype=torch.int64, device=self.device),
                    torch.full((instr_feat.shape[0],), 2, dtype=torch.int64, device=self.device),
                ],
                dim=0,
            )
            attn = torch.ones(features.shape[0], dtype=torch.bool, device=self.device)
            batch_features.append(features)
            batch_attn.append(attn)
            batch_token_type.append(token_type)
        max_len = max(x.shape[0] for x in batch_features)
        dim = batch_features[0].shape[-1]
        padded_features = torch.zeros(len(batch_features), max_len, dim, dtype=batch_features[0].dtype, device=self.device)
        padded_attn = torch.zeros(len(batch_features), max_len, dtype=torch.bool, device=self.device)
        padded_token_type = torch.zeros(len(batch_features), max_len, dtype=torch.long, device=self.device)
        for index, features in enumerate(batch_features):
            n = features.shape[0]
            padded_features[index, :n] = features
            padded_attn[index, :n] = batch_attn[index]
            padded_token_type[index, :n] = batch_token_type[index]
        return padded_features.float(), padded_attn, padded_token_type
