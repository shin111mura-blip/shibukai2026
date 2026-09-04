from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DepthFeatureInfo:
    depth_shape: tuple[int, ...]
    finite_fraction: float
    depth_min: float | None
    depth_max: float | None
    depth_mean: float | None
    depth_std: float | None
    feature_dim: int


def ensure_depth_bchw(depth: Any) -> "np.ndarray":
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, None, :, :]
    elif arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr.transpose(2, 0, 1)[None, ...]
        else:
            arr = arr[:, None, :, :]
    elif arr.ndim == 4:
        pass
    else:
        raise ValueError(f"Expected depth with 2, 3, or 4 dims, got shape {arr.shape}")
    if arr.shape[1] != 1:
        raise ValueError(f"Expected depth channel dimension 1, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    return arr


def depth_to_feature(depth: Any, grid: int = 16) -> tuple[np.ndarray, DepthFeatureInfo]:
    arr = ensure_depth_bchw(depth)[0, 0]
    finite = np.isfinite(arr)
    clean = np.where(finite, arr, 0.0).astype(np.float32)
    h, w = clean.shape
    if h % grid or w % grid:
        raise ValueError(f"Depth shape {clean.shape} is not divisible by grid={grid}")
    pooled = clean.reshape(grid, h // grid, grid, w // grid).mean(axis=(1, 3)).astype(np.float32)
    mean = float(pooled.mean())
    std = float(pooled.std())
    spatial = (pooled - mean) / std if std > 1e-6 else pooled - mean
    values = clean[finite]
    stats = np.array(
        [
            float(values.mean()) if values.size else 0.0,
            float(values.std()) if values.size else 0.0,
            float(values.min()) if values.size else 0.0,
            float(values.max()) if values.size else 0.0,
            float(np.percentile(values, 10)) if values.size else 0.0,
            float(np.percentile(values, 50)) if values.size else 0.0,
            float(np.percentile(values, 90)) if values.size else 0.0,
            float(finite.mean()),
        ],
        dtype=np.float32,
    )
    feature = np.concatenate([spatial.reshape(-1), stats], axis=0).astype(np.float32)
    info = DepthFeatureInfo(
        depth_shape=tuple(arr.shape),
        finite_fraction=float(finite.mean()),
        depth_min=float(values.min()) if values.size else None,
        depth_max=float(values.max()) if values.size else None,
        depth_mean=float(values.mean()) if values.size else None,
        depth_std=float(values.std()) if values.size else None,
        feature_dim=int(feature.shape[0]),
    )
    return feature, info
