from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import CONDITION_SPECS, LOCKED_MANIFESTS, REPORTS, assert_lora_overrides_match_locked, manifest_checksum, write_json
from .diagnostics import run_locked_spec_checks, run_manifest_checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenVLA depth-free graph teacher training entrypoint.")
    parser.add_argument("--condition", choices=sorted(CONDITION_SPECS), required=True)
    parser.add_argument("--seed", type=int, choices=sorted(LOCKED_MANIFESTS), required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs/depthfree_graph_teacher"))
    parser.add_argument("--lambda-graph", type=float, default=0.1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-openvla-full-train", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    condition = CONDITION_SPECS[args.condition]
    manifest = LOCKED_MANIFESTS[args.seed]
    lora_overrides = {
        "use_lora": True,
        "lora_rank": 32,
        "lora_dropout": 0.0,
        "use_quantization": False,
        "learning_rate": 0.0005,
        "batch_size": 8,
        "grad_accumulation_steps": 2,
        "max_steps": 10000,
        "save_steps": 1000,
        "image_aug": True,
        "seed": 42,
        "dataset_name": "libero_spatial_no_noops",
    }
    assert_lora_overrides_match_locked(lora_overrides)
    spec_checks = run_locked_spec_checks()
    manifest_checks = run_manifest_checks()[str(args.seed)]
    run_dir = args.run_root / args.condition / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "uses_depth": condition["uses_depth"],
        "uses_graph_aux": condition["uses_graph_aux"],
        "uses_action_loss": condition.get("uses_action_loss", True),
        "lambda_graph": args.lambda_graph if condition["uses_graph_aux"] else 0.0,
        "manifest": str(manifest),
        "manifest_checksum": manifest_checksum(manifest),
        "locked_lora_overrides": lora_overrides,
        "locked_graph_teacher": spec_checks,
        "manifest_checks": manifest_checks,
        "status": "preflight_passed",
    }
    write_json(run_dir / "run_metadata.json", metadata)
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", "run_dir": str(run_dir)}, sort_keys=True))
        return
    if not args.allow_openvla_full_train:
        metadata["status"] = "blocked_before_full_train"
        metadata["blocker"] = (
            "Full OpenVLA RLDS training requires the sidecar sample_key/depth/graph batch hook to be wired into "
            "the local OpenVLA dataloader. Use --allow-openvla-full-train only after that hook is verified."
        )
        write_json(run_dir / "run_metadata.json", metadata)
        raise SystemExit(metadata["blocker"])
    raise SystemExit("Full OpenVLA training launch is intentionally gated until local dataloader hook verification passes.")


if __name__ == "__main__":
    main()
