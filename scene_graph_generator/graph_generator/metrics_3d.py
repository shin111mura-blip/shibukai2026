from __future__ import annotations

from typing import Dict

import numpy as np


def xyz_metrics(pred_xyz: np.ndarray, gt_xyz: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    valid = mask.astype(bool)
    if not valid.any():
        return {"num_points": 0, "mae": 0.0, "rmse": 0.0, "median_l2": 0.0, "within_2cm": 0.0, "within_5cm": 0.0, "within_10cm": 0.0}
    delta = pred_xyz[valid] - gt_xyz[valid]
    l2 = np.linalg.norm(delta, axis=-1)
    return {
        "num_points": int(valid.sum()),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "median_l2": float(np.median(l2)),
        "mean_l2": float(np.mean(l2)),
        "within_2cm": float(np.mean(l2 <= 0.02)),
        "within_5cm": float(np.mean(l2 <= 0.05)),
        "within_10cm": float(np.mean(l2 <= 0.10)),
    }
