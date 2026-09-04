#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()


def command_output(cmd: list[str]) -> dict:
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10)
        return {"ok": True, "output": out.strip()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    report = {"nvidia_smi": command_output(["nvidia-smi"]) if shutil.which("nvidia-smi") else {"ok": False, "error": "nvidia-smi not found"}}
    try:
        import torch

        report["torch"] = {"version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "cuda_device_count": int(torch.cuda.device_count())}
    except Exception as exc:
        report["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        import transformers

        report["transformers"] = {
            "version": getattr(transformers, "__version__", "unknown"),
            "has_Qwen3VLForConditionalGeneration": hasattr(transformers, "Qwen3VLForConditionalGeneration"),
            "has_AutoModelForMultimodalLM": hasattr(transformers, "AutoModelForMultimodalLM"),
        }
    except Exception as exc:
        report["transformers"] = {"error": f"{type(exc).__name__}: {exc}"}
    report["hf_cache"] = {
        "root": "outputs/scene_graph_gt_v0/hf_cache",
        "exists": Path("outputs/scene_graph_gt_v0/hf_cache").exists(),
    }
    out = Path("outputs/scene_graph_gt_v0/reports/qwen_runtime_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

