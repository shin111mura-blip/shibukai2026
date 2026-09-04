from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import BUNDLE_INDEX, CONDITION_SPECS, LOCKED_MANIFESTS, ROOT, read_json, resolve_workspace_path
from .sidecar_dataset import build_sidecar_index, graph_target_from_row, load_bundle_rows, load_depth_tensor_store


def _scalar_int(value: Any) -> int:
    arr = np.asarray(value)
    return int(arr.reshape(-1)[0])


@dataclass(frozen=True)
class RLDSSidecarConfig:
    condition: str
    manifest_path: Path
    index_path: Path = BUNDLE_INDEX
    episode_map_path: Path | None = None
    require_depth: bool = False
    require_graph: bool = False


class RLDSSidecarLookup:
    """Strict sample_key/depth/graph target lookup for OpenVLA RLDS transitions."""

    def __init__(self, cfg: RLDSSidecarConfig):
        if cfg.condition not in CONDITION_SPECS:
            raise ValueError(f"Unknown graph internalization condition: {cfg.condition}")
        self.cfg = cfg
        self.condition_spec = CONDITION_SPECS[cfg.condition]
        self.manifest = read_json(cfg.manifest_path)
        self.allowed_episodes = set(int(x) for x in self.manifest["selected_global_episode_indices"])
        self.tfds_to_sidecar_episode = self._load_episode_map(cfg.episode_map_path)
        rows = load_bundle_rows(cfg.index_path)
        self.by_episode_step: dict[tuple[int, int], dict[str, Any]] = {}
        self.by_sample_key: dict[str, dict[str, Any]] = {}
        duplicates = []
        for row in rows:
            key = (int(row["global_episode_index"]), int(row["step_id"]))
            if key in self.by_episode_step:
                duplicates.append(key)
            self.by_episode_step[key] = row
            self.by_sample_key[row["sample_key"]] = row
        if duplicates:
            raise ValueError(f"Duplicate sidecar episode/step keys: {duplicates[:5]}")
        self.selected_sample_keys = {
            row["sample_key"]
            for row in rows
            if int(row["global_episode_index"]) in self.allowed_episodes
        }
        self.depth_tensors = load_depth_tensor_store() if cfg.require_graph else None
        self.lookup_count = 0
        self.lookup_seconds = 0.0

    @classmethod
    def from_env(cls) -> "RLDSSidecarLookup | None":
        enabled = os.environ.get("OPENVLA_GRAPH_INTERNALIZATION", "").lower() in {"1", "true", "yes"}
        if not enabled:
            return None
        condition = os.environ.get("OPENVLA_GRAPH_CONDITION")
        if not condition:
            raise ValueError("OPENVLA_GRAPH_INTERNALIZATION requires OPENVLA_GRAPH_CONDITION.")
        manifest = os.environ.get("OPENVLA_GRAPH_MANIFEST_JSON") or os.environ.get("OPENVLA_RLDS_DEMO_SPLIT_JSON")
        if not manifest:
            raise ValueError("OPENVLA_GRAPH_INTERNALIZATION requires OPENVLA_GRAPH_MANIFEST_JSON or OPENVLA_RLDS_DEMO_SPLIT_JSON.")
        index = resolve_workspace_path(os.environ.get("OPENVLA_GRAPH_BUNDLE_INDEX", str(BUNDLE_INDEX)))
        spec = CONDITION_SPECS[condition]
        return cls(
            RLDSSidecarConfig(
                condition=condition,
                manifest_path=resolve_workspace_path(manifest),
                index_path=index,
                episode_map_path=resolve_workspace_path(os.environ["OPENVLA_RLDS_EPISODE_MAP_JSON"])
                if os.environ.get("OPENVLA_RLDS_EPISODE_MAP_JSON")
                else None,
                require_depth=spec["uses_depth"],
                require_graph=spec["uses_graph_aux"],
            )
        )

    @staticmethod
    def _load_episode_map(path: Path | None) -> dict[int, int]:
        if path is None:
            return {}
        payload = read_json(path)
        mapping = payload.get("tfds_to_sidecar_global_episode_index", {})
        return {int(k): int(v) for k, v in mapping.items()}

    def sidecar_episode_for_rlds_episode(self, episode_index: int) -> int:
        episode_index = int(episode_index)
        if self.tfds_to_sidecar_episode:
            try:
                return self.tfds_to_sidecar_episode[episode_index]
            except KeyError as exc:
                raise KeyError(
                    f"Missing TFDS->sidecar episode mapping for tfds_episode_index={episode_index}. "
                    f"Regenerate {self.cfg.episode_map_path}."
                ) from exc
        return episode_index

    def stable_key_for_transition(self, episode_index: int, timestep: int) -> str:
        sidecar_episode_index = self.sidecar_episode_for_rlds_episode(episode_index)
        row = self.by_episode_step.get((sidecar_episode_index, int(timestep)))
        if row is None:
            raise KeyError(
                f"Missing sidecar row for sidecar_episode_index={sidecar_episode_index} "
                f"(tfds_episode_index={episode_index}), timestep={timestep}"
            )
        sample_key = row["sample_key"]
        if sample_key not in self.selected_sample_keys:
            raise KeyError(f"Sample {sample_key} is outside locked manifest {self.cfg.manifest_path}")
        return sample_key

    def lookup(self, rlds_batch: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        obs = rlds_batch["observation"]
        if "episode_index" not in obs or "timestep" not in obs:
            raise KeyError("RLDS sidecar lookup requires observation['episode_index'] and observation['timestep'].")
        episode_index = _scalar_int(obs["episode_index"][0])
        timestep = _scalar_int(obs["timestep"][0])
        sidecar_episode_index = self.sidecar_episode_for_rlds_episode(episode_index)
        sample_key = self.stable_key_for_transition(episode_index, timestep)
        row = self.by_sample_key[sample_key]
        out: dict[str, Any] = {
            "sample_key": sample_key,
            "global_episode_index": sidecar_episode_index,
            "rlds_episode_index": episode_index,
            "timestep": timestep,
            "task_id": int(row["task_id"]),
        }
        if self.cfg.require_depth:
            raise RuntimeError(
                "Depth condition requested, but the local libero_spatial_no_noops RLDS dataset has no raw depth "
                "field and the locked sidecar contains only 264-D depth features for graph xyz targets. "
                "Refusing to synthesize pseudo-depth for the vision adapter."
            )
        if self.cfg.require_graph:
            target = graph_target_from_row(row, self.depth_tensors)
            out["graph_targets"] = {
                "y_node": torch.from_numpy(target["y_node"]),
                "y_edge": torch.from_numpy(target["y_edge"]),
                "y_xyz": torch.from_numpy(target["y_xyz"]),
                "y_xyz_mask": torch.from_numpy(target["y_xyz_mask"]),
            }
            out["graph_masks"] = {
                "y_xyz_mask": torch.from_numpy(target["y_xyz_mask"]),
            }
        self.lookup_count += 1
        self.lookup_seconds += time.perf_counter() - start
        out["sidecar_lookup_seconds"] = self.lookup_seconds / max(1, self.lookup_count)
        return out


def manifest_path_for_seed(seed: int) -> Path:
    return LOCKED_MANIFESTS[int(seed)]
