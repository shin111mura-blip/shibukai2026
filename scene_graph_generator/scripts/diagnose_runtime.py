#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


OUTPUT_ROOT = Path("outputs/scene_graph_generator_openvla_spatial")


def run_command(cmd: List[str], timeout: int = 120) -> Dict:
    started = time.time()
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return {
            "command": cmd,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_sec": round(time.time() - started, 3),
        }
    except FileNotFoundError as exc:
        return {
            "command": cmd,
            "exit_code": 127,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_sec": round(time.time() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": cmd,
            "exit_code": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"timeout after {timeout}s",
            "elapsed_sec": round(time.time() - started, 3),
        }


def append_command_markdown(lines: List[str], result: Dict) -> None:
    lines.append(f"## `$ {' '.join(result['command'])}`")
    lines.append("")
    lines.append(f"- Exit code: `{result['exit_code']}`")
    lines.append(f"- Elapsed: `{result['elapsed_sec']}s`")
    lines.append("")
    lines.append("stdout:")
    lines.append("```")
    lines.append(result["stdout"].strip())
    lines.append("```")
    lines.append("")
    lines.append("stderr:")
    lines.append("```")
    lines.append(result["stderr"].strip())
    lines.append("```")
    lines.append("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = ap.parse_args()
    report_dir = args.output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        ["git", "status", "--short"],
        ["git", "diff", "--stat"],
        ["docker", "--version"],
        ["docker", "compose", "version"],
        ["docker", "info"],
        ["docker", "info", "--format", "{{json .Runtimes}}"],
        ["docker", "ps", "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}"],
        ["nvidia-smi"],
        ["command", "-v", "nvidia-ctk"],
        ["command", "-v", "nvidia-container-cli"],
    ]
    # `command -v` is shell builtin; use `sh -lc` only for these two simple probes.
    normalized = []
    for cmd in commands:
        if cmd[:2] == ["command", "-v"]:
            normalized.append(["sh", "-lc", " ".join(cmd)])
        else:
            normalized.append(cmd)

    results = [run_command(cmd) for cmd in normalized]

    image_result = run_command(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"])
    results.append(image_result)
    cuda_image = None
    for line in image_result["stdout"].splitlines():
        if line.startswith("nvidia/cuda:"):
            cuda_image = line.strip()
            break
    if cuda_image:
        results.append(run_command(["docker", "run", "--rm", "--gpus", "all", cuda_image, "nvidia-smi"], timeout=180))
    else:
        results.append(
            {
                "command": ["docker", "run", "--rm", "--gpus", "all", "<existing-cuda-image>", "nvidia-smi"],
                "exit_code": 125,
                "stdout": "",
                "stderr": "No local nvidia/cuda image was found; skipped to avoid network pull.",
                "elapsed_sec": 0,
            }
        )

    state_lines = [
        "# Runtime Resolution Initial State",
        "",
        f"- Timestamp: `{time.strftime('%Y-%m-%d %H:%M:%S %z')}`",
        f"- Host: `{platform.node()}`",
        f"- Platform: `{platform.platform()}`",
        f"- Python: `{sys.version}`",
        f"- Working directory: `{Path.cwd()}`",
        f"- Final report backup: `reports/final_report.md.bak_runtime_resolution_20260718`",
        "",
    ]
    for result in results[:2]:
        append_command_markdown(state_lines, result)
    (report_dir / "runtime_resolution_initial_state.md").write_text("\n".join(state_lines))

    cmd_lines: List[str] = []
    err_lines: List[str] = []
    gpu_lines = ["# Docker GPU Diagnosis", ""]
    for result in results:
        cmd_lines.append(json.dumps(result, sort_keys=True))
        if result["exit_code"] != 0 or result["stderr"].strip():
            err_lines.append(json.dumps(result, sort_keys=True))
        append_command_markdown(gpu_lines, result)

    (report_dir / "runtime_resolution_commands.log").write_text("\n".join(cmd_lines) + "\n")
    (report_dir / "runtime_resolution_errors.log").write_text("\n".join(err_lines) + ("\n" if err_lines else ""))
    (report_dir / "docker_gpu_diagnosis.md").write_text("\n".join(gpu_lines))

    summary = {
        "nvidia_smi_exit_code": next((r["exit_code"] for r in results if r["command"] == ["nvidia-smi"]), None),
        "docker_gpu_test_exit_code": results[-1]["exit_code"],
        "nvidia_ctk": shutil.which("nvidia-ctk"),
        "nvidia_container_cli": shutil.which("nvidia-container-cli"),
        "docker_available": any(r["command"] == ["docker", "--version"] and r["exit_code"] == 0 for r in results),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
