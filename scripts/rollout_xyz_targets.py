#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


class RolloutXyzTargetCache:
    """Resolve rollout 3D position targets from sidecars, falling back to frames.npz."""

    def __init__(self, ontology: dict[str, Any], *, data_root: Path | None = None, sidecar_root: Path | None = None):
        self.node_order = [node_id for node_id, meta in sorted(ontology["nodes"].items(), key=lambda item: item[1]["index"])]
        self.data_root = data_root
        self.sidecar_root = sidecar_root
        self.cache: dict[str, dict[str, Any] | None] = {}
        self.sidecar_index: dict[str, Path] | None = None

    def _episode_suffix(self, episode_dir: Path) -> Path | None:
        parts = episode_dir.parts
        if "episodes" not in parts:
            return None
        return Path(*parts[parts.index("episodes") + 1 :])

    def _build_sidecar_index(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        if self.sidecar_root is not None and self.sidecar_root.exists():
            for path in self.sidecar_root.rglob("graph3d_positions.json"):
                episode_id = path.parent.name
                index.setdefault(episode_id, path)
        return index

    def sidecar_path_for_episode(self, episode_dir: Path) -> Path | None:
        direct = episode_dir / "graph3d_positions.json"
        if direct.exists():
            return direct
        if self.sidecar_root is None:
            return None

        suffix = self._episode_suffix(episode_dir)
        candidates = []
        if suffix is not None:
            candidates.extend(
                [
                    self.sidecar_root / suffix / "graph3d_positions.json",
                    self.sidecar_root / "episodes" / suffix / "graph3d_positions.json",
                ]
            )
        candidates.append(self.sidecar_root / episode_dir.name / "graph3d_positions.json")
        for candidate in candidates:
            if candidate.exists():
                return candidate

        if self.sidecar_index is None:
            self.sidecar_index = self._build_sidecar_index()
        return self.sidecar_index.get(episode_dir.name)

    def _get_sidecar(self, episode_dir: Path) -> dict[str, Any] | None:
        key = str(episode_dir)
        if key not in self.cache:
            sidecar = self.sidecar_path_for_episode(episode_dir)
            self.cache[key] = read_json(sidecar) if sidecar is not None and sidecar.exists() else None
        return self.cache[key]

    def get(self, episode_dir: Path, arrays: Any, frame_index: int) -> tuple[np.ndarray, np.ndarray, str]:
        sidecar = self._get_sidecar(episode_dir)
        if sidecar is None:
            return (
                arrays["position_target"][frame_index].astype(np.float32),
                arrays["position_valid_mask"][frame_index].astype(np.float32),
                "frames_npz",
            )
        records = sidecar.get("position_records") or []
        if frame_index >= len(records):
            return (
                arrays["position_target"][frame_index].astype(np.float32),
                arrays["position_valid_mask"][frame_index].astype(np.float32),
                "frames_npz",
            )
        world_positions = records[frame_index].get("world_positions") or {}
        xyz = np.zeros((len(self.node_order), 3), dtype=np.float32)
        mask = np.zeros((len(self.node_order),), dtype=np.float32)
        for idx, node_id in enumerate(self.node_order):
            value = world_positions.get(node_id)
            if value is None:
                continue
            xyz[idx] = np.asarray(value, dtype=np.float32).reshape(3)
            mask[idx] = 1.0
        return xyz, mask, "graph3d_positions"
