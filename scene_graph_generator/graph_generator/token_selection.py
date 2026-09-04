from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TokenLayout:
    bos_positions: List[int]
    image_start: int
    image_end: int
    instruction_positions: List[int]
    padding_positions: List[int]


def infer_prismatic_token_layout(input_ids, attention_mask, image_token_count: int, tokenizer: Optional[Any] = None) -> TokenLayout:
    ids = input_ids[0].detach().cpu().tolist() if hasattr(input_ids, "detach") else list(input_ids[0])
    mask = attention_mask[0].detach().cpu().tolist() if hasattr(attention_mask, "detach") else list(attention_mask[0])
    bos_id = getattr(tokenizer, "bos_token_id", None) if tokenizer is not None else None
    bos_positions = [idx for idx, tok in enumerate(ids) if bos_id is not None and tok == bos_id]
    if not bos_positions:
        bos_positions = [0]
    image_start = 1
    image_end = image_start + int(image_token_count)
    instruction_positions = []
    padding_positions = []
    for idx, is_attended in enumerate(mask):
        if not is_attended:
            padding_positions.append(idx)
            continue
        if idx in bos_positions:
            continue
        # Hidden states include projected image tokens after BOS, so original token idx>=1 shifts by image count.
        instruction_positions.append(idx + image_token_count)
    return TokenLayout(
        bos_positions=bos_positions,
        image_start=image_start,
        image_end=image_end,
        instruction_positions=instruction_positions,
        padding_positions=padding_positions,
    )

