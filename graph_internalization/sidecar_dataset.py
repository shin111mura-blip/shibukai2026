from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import BUNDLE_INDEX, DEPTH_FEATURE_DIR, ONTOLOGY_PATH, ROOT, read_json, read_jsonl, sha256_payload


@dataclass(frozen=True)
class ManifestSelection:
    path: Path
    checksum: str
    seed: int
    sample_keys: tuple[str, ...]
    selected_global_episode_indices: tuple[int, ...]
    overlap_with_graph_generator_train: int


def stable_key(row: dict[str, Any]) -> str:
    return str(row["sample_key"])


def load_bundle_rows(path: Path = BUNDLE_INDEX) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        key = stable_key(row)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"Duplicate sample_key rows in bundle index: {duplicates[:5]}")
    return rows


def _sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    return (int(row["task_id"]), int(row["global_episode_index"]), int(row["step_id"]), stable_key(row))


def manifest_selection(manifest_path: Path, rows: Iterable[dict[str, Any]] | None = None) -> ManifestSelection:
    manifest = read_json(manifest_path)
    source_rows = list(load_bundle_rows() if rows is None else rows)
    episodes = set(int(x) for x in manifest["selected_global_episode_indices"])
    selected = [row for row in source_rows if int(row["global_episode_index"]) in episodes]
    if len(selected) != int(manifest["selected_steps"]):
        raise ValueError(
            f"Manifest {manifest_path} selected_steps mismatch: expected {manifest['selected_steps']} got {len(selected)}"
        )
    keys = tuple(stable_key(row) for row in sorted(selected, key=_sort_key))
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate sample keys after manifest filtering: {manifest_path}")
    return ManifestSelection(
        path=manifest_path,
        checksum=str(manifest["checksum"]),
        seed=int(manifest["seed"]),
        sample_keys=keys,
        selected_global_episode_indices=tuple(sorted(episodes)),
        overlap_with_graph_generator_train=int(manifest["overlap_with_graph_generator_train"]),
    )


def assert_same_keys_for_conditions(selection: ManifestSelection, condition_names: Iterable[str]) -> dict[str, Any]:
    keys_by_condition = {condition: selection.sample_keys for condition in condition_names}
    reference = next(iter(keys_by_condition.values()))
    for condition, keys in keys_by_condition.items():
        if keys != reference:
            raise AssertionError(f"Sample key mismatch for {condition}")
    return {
        "conditions": sorted(keys_by_condition),
        "sample_count": len(reference),
        "manifest_checksum": selection.checksum,
        "key_set_sha256": sha256_payload(list(reference)),
    }


def build_sidecar_index(rows: Iterable[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    out = {}
    for row in load_bundle_rows() if rows is None else rows:
        key = stable_key(row)
        if key in out:
            raise ValueError(f"Duplicate sample key: {key}")
        out[key] = row
    return out


def graph_target_from_row(row: dict[str, Any], depth_tensors: dict[str, Any] | None = None) -> dict[str, Any]:
    from scene_graph_generator.graph_generator.schema import compact_graph
    from scene_graph_generator.graph_generator.targets import encode_targets

    ontology = read_json(ONTOLOGY_PATH)
    graph_path = ROOT / row["graph_reference"]
    graph = compact_graph(read_json(graph_path))
    y_node, y_edge = encode_targets(graph, ontology)
    target = {
        "y_node": np.asarray(y_node, dtype=np.float32),
        "y_edge": np.asarray(y_edge, dtype=np.float32),
        "graph": graph,
    }
    if depth_tensors is not None:
        legacy = row["legacy_sample_key"]
        target["y_xyz"] = depth_tensors[f"{legacy}__xyz_target"].detach().cpu().numpy().astype(np.float32)
        target["y_xyz_mask"] = depth_tensors[f"{legacy}__xyz_mask"].detach().cpu().numpy().astype(np.float32)
    return target


def load_depth_tensor_store() -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(DEPTH_FEATURE_DIR / "depth_features.safetensors"), device="cpu")


def read_jsonl_stream(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
