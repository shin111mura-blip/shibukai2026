from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MANIFESTS = ROOT / "manifests"
BUNDLE_INDEX = ROOT / "artifacts/openvla_graph_internalization_bundle_v2/index.jsonl"
ONTOLOGY_PATH = ROOT / "outputs/scene_graph_generator_openvla_spatial/ontology/ontology.json"
FEATURE_CACHE_DIR = ROOT / "outputs/scene_graph_generator_openvla_spatial/feature_cache/all_frames"
DEPTH_FEATURE_DIR = ROOT / "outputs/scene_graph_generator_openvla_spatial/depth_features/all_frames"

PRIMARY_LOCK_PATH = REPORTS / "primary_graph_teacher.lock.json"
GRAPH_SPEC_PATH = REPORTS / "depth_free_graph_specification.json"
GRAPH_RELATION_PATH = REPORTS / "depth_free_graph_relation_vocabulary.json"
LORA_SPEC_PATH = REPORTS / "openvla_lora_locked_spec.yaml"
DEPTH_SPEC_PATH = REPORTS / "current_depth_specification.json"
HOST_ROOT_PREFIX = "/home/user/Desktop/HRI2027"

GRAPH_TEACHER_CLASS = "OpenVLAOnlyPooledMLP3DGraphGenerator"
GRAPH_TEACHER_ARCH = os.environ.get("GRAPH_TEACHER_ARCH", "pooled_mlp_openvla_3d")
GRAPH_TEACHER_SHA256 = os.environ.get(
    "GRAPH_TEACHER_SHA256",
    "7b8e3cbd5cdf78de6d3c67d2f4ad6fd078469ebecbb0c6ce97475110db2eade6",
)
RELATION_VOCAB_SHA256 = "c9fe82fc35570f3972934f4eaae104e707f8c4a0de39fd457caa9729417dc5c4"

CONDITION_SPECS: dict[str, dict[str, bool]] = {
    "rgb_action": {"uses_depth": False, "uses_graph_aux": False, "uses_action_loss": True},
    "rgbd_action": {"uses_depth": True, "uses_graph_aux": False, "uses_action_loss": True},
    "rgb_graph": {"uses_depth": False, "uses_graph_aux": True, "uses_action_loss": True},
    "rgb_graph_no_action": {"uses_depth": False, "uses_graph_aux": True, "uses_action_loss": False},
    "rgbd_graph": {"uses_depth": True, "uses_graph_aux": True, "uses_action_loss": True},
}

LOCKED_MANIFESTS = {
    101: MANIFESTS / "depth_free_teacher_holdout_seed101.json",
    202: MANIFESTS / "depth_free_teacher_holdout_seed202.json",
    303: MANIFESTS / "depth_free_teacher_holdout_seed303.json",
    404: MANIFESTS / "depth_free_teacher_holdout_seed404.json",
    505: MANIFESTS / "depth_free_teacher_holdout_seed505.json",
}


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_payload(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def resolve_workspace_path(path: str | Path) -> Path:
    p = Path(path)
    text = str(p)
    if text.startswith(HOST_ROOT_PREFIX):
        return ROOT / text[len(HOST_ROOT_PREFIX) :].lstrip("/")
    if text.startswith("/workspace"):
        return ROOT / text[len("/workspace") :].lstrip("/")
    return p


@dataclass(frozen=True)
class LockedGraphTeacherSpec:
    primary_teacher: str
    architecture: str
    class_name: str
    checkpoint: Path
    checkpoint_sha256: str
    relation_vocabulary_sha256: str
    openvla_dim: int
    num_nodes: int
    num_predicates: int
    hidden_dim: int
    num_layers: int
    dropout: float
    feature_layer: int
    image_token_type: int
    instruction_token_type: int
    edge_pos_weight: list[float]
    xyz_weight: float


def load_locked_graph_teacher_spec() -> LockedGraphTeacherSpec:
    lock = read_json(PRIMARY_LOCK_PATH)
    spec = read_json(GRAPH_SPEC_PATH)
    ontology = read_json(ONTOLOGY_PATH)
    loss_weights = spec["loss"]["internal_weights"]["edge_pos_weight"]["edge_pos_weight_vector_by_predicate_id"]
    return LockedGraphTeacherSpec(
        primary_teacher=lock["PRIMARY_GRAPH_TEACHER"],
        architecture=GRAPH_TEACHER_ARCH,
        class_name=spec["architecture"]["class_name"],
        checkpoint=resolve_workspace_path(os.environ.get("GRAPH_TEACHER_CHECKPOINT_PATH", spec["checkpoint"]["path"])),
        checkpoint_sha256=GRAPH_TEACHER_SHA256,
        relation_vocabulary_sha256=lock["relation_vocabulary_sha256"],
        openvla_dim=int(spec["architecture"]["input_dimensions"]["openvla_dim"]),
        num_nodes=len(ontology["nodes"]),
        num_predicates=len(ontology["predicates"]),
        hidden_dim=int(spec["architecture"]["hidden_dim"]),
        num_layers=int(spec["architecture"]["num_layers"]),
        dropout=float(spec["architecture"]["dropout"]),
        feature_layer=int(spec["feature_source"]["layer_index"]),
        image_token_type=int(spec["feature_source"]["token_type_mask"]["image"]),
        instruction_token_type=int(spec["feature_source"]["token_type_mask"]["instruction"]),
        edge_pos_weight=[float(x) for x in loss_weights],
        xyz_weight=float(spec["loss"]["internal_weights"]["xyz_weight"]),
    )


def assert_locked_graph_teacher_files() -> dict[str, Any]:
    spec = load_locked_graph_teacher_spec()
    failures: list[str] = []
    if spec.primary_teacher != "depth_free":
        failures.append(f"PRIMARY_GRAPH_TEACHER={spec.primary_teacher}")
    if spec.class_name != GRAPH_TEACHER_CLASS:
        failures.append(f"class={spec.class_name}")
    if sha256_file(spec.checkpoint) != GRAPH_TEACHER_SHA256:
        failures.append("checkpoint sha256 mismatch")
    if spec.checkpoint_sha256 != GRAPH_TEACHER_SHA256:
        failures.append("lock checkpoint sha256 mismatch")
    if spec.relation_vocabulary_sha256 != RELATION_VOCAB_SHA256:
        failures.append("lock relation sha256 mismatch")
    if failures:
        raise AssertionError("; ".join(failures))
    return {
        "primary_teacher": spec.primary_teacher,
        "class_name": spec.class_name,
        "checkpoint": str(spec.checkpoint),
        "checkpoint_sha256": spec.checkpoint_sha256,
        "relation_vocabulary_sha256": spec.relation_vocabulary_sha256,
        "num_nodes": spec.num_nodes,
        "num_predicates": spec.num_predicates,
    }


def load_lora_spec() -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read the locked LoRA spec") from exc
    with open(LORA_SPEC_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def assert_lora_overrides_match_locked(overrides: Mapping[str, Any]) -> None:
    spec = load_lora_spec()
    expected = {
        "vla_path": spec["base_model"]["name"],
        "use_lora": spec["lora"]["enabled"],
        "lora_rank": spec["lora"]["rank"],
        "lora_dropout": spec["lora"]["dropout"],
        "use_quantization": spec["base_model"]["quantization"],
        "learning_rate": spec["training"]["learning_rate"],
        "batch_size": spec["training"]["batch_size_per_device"],
        "grad_accumulation_steps": spec["training"]["gradient_accumulation_steps"],
        "max_steps": spec["training"]["max_steps"],
        "save_steps": spec["training"]["save_steps"],
        "image_aug": spec["training"]["image_aug"],
        "seed": spec["training"]["seed"],
        "dataset_name": spec["dataset"]["rlds_dataset_name"],
    }
    mismatches = []
    for key, value in expected.items():
        if key in overrides and overrides[key] != value:
            mismatches.append(f"{key}: got {overrides[key]!r}, expected {value!r}")
    if mismatches:
        raise AssertionError("Locked LoRA spec mismatch: " + "; ".join(mismatches))


def manifest_checksum(path: Path) -> str:
    return str(read_json(path)["checksum"])
