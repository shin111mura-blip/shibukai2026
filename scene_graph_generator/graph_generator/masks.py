from __future__ import annotations

from typing import Any, Dict

import numpy as np


def relation_validity_mask(ontology: Dict[str, Any]) -> np.ndarray:
    nodes = ontology["nodes"]
    preds = ontology["predicates"]
    k = len(nodes)
    r = len(preds)
    mask = np.ones((k, k, r), dtype=bool)
    for i in range(k):
        mask[i, i, :] = False
    if "grasping" in preds:
        g = preds["grasping"]
        idx_to_type = {meta["index"]: meta["entity_type"] for meta in nodes.values()}
        for i in range(k):
            for j in range(k):
                mask[i, j, g] = idx_to_type[i] == "gripper" and idx_to_type[j] == "object" and i != j
    return mask

