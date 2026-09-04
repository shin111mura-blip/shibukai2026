#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from rollout_collection_common import (
    DATA_ROOT,
    JOBS_FULL,
    POLICY_POOL_CONFIG,
    ROOT,
    SCHEMA_LOCK_JSON,
    add_runtime_paths,
    atomic_write_episode,
    checksum_episode_files,
    classify_failure,
    encode_graph_targets,
    load_config,
    read_json,
    read_jsonl,
    resolve_policy_runtime,
    validate_episode_dir,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect OpenVLA rollout frames with pre-action oracle graphs.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "rollout_collection_libero_spatial_full.yaml")
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--policy-config", type=Path, default=POLICY_POOL_CONFIG)
    parser.add_argument("--job-manifest", type=Path, default=JOBS_FULL)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=None)
    parser.add_argument("--maximum-steps", type=int, default=None)
    parser.add_argument("--initial-state-ids", type=int, nargs="*", default=None)
    parser.add_argument("--rollout-seed", type=int, default=None)
    parser.add_argument("--save-terminal-frame", action="store_true", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--full-collection", action="store_true")
    parser.add_argument("--image-compression", choices=["npz", "png"], default="npz")
    parser.add_argument("--failure-classification", choices=["heuristic", "none"], default="heuristic")
    parser.add_argument("--validate-after-episode", action="store_true")
    parser.add_argument("--split-seed", type=int, default=20260803)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def get_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs = read_jsonl(args.job_manifest)
    wanted_phase = "preflight" if args.preflight else ("full" if args.full_collection else None)
    selected = []
    for job in jobs:
        if job["policy_id"] != args.policy_id:
            continue
        if int(job["assigned_worker"]) != args.worker_id:
            continue
        if wanted_phase and job.get("phase") != wanted_phase:
            continue
        if args.initial_state_ids is not None and int(job["initial_state_id"]) not in set(args.initial_state_ids):
            continue
        selected.append(job)
    if args.episodes_per_task is not None:
        by_task = Counter()
        limited = []
        for job in selected:
            task = int(job["task_id"])
            if by_task[task] < args.episodes_per_task:
                limited.append(job)
                by_task[task] += 1
        selected = limited
    return selected


def build_env_for_task(task_id: int, suite_name: str, image_size: int):
    add_runtime_paths()
    from libero.libero import benchmark, get_libero_path
    from scene_graph.rule_generator import create_env

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = create_env(str(bddl_file), image_size)
    return env, task, task_suite.get_task_init_states(task_id), bddl_file


def load_policy(runtime):
    add_runtime_paths()
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model

    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(runtime.checkpoint),
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=runtime.center_crop,
        unnorm_key=runtime.unnorm_key,
    )
    model = get_model(cfg)
    processor = get_processor(cfg)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    if cfg.unnorm_key not in model.norm_stats:
        raise RuntimeError(f"unnorm_key missing from model.norm_stats: {cfg.unnorm_key}")
    return cfg, model, processor


def collect_one_job(job: dict[str, Any], args: argparse.Namespace, config: dict[str, Any], policy_runtime, model_bundle, schema: dict[str, Any]) -> Path:
    add_runtime_paths()
    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_image, quat2axisangle
    from experiments.robot.robot_utils import get_action, invert_gripper_action, normalize_gripper_action, set_seed_everywhere
    from scene_graph.rule_generator import build_frame_graphs, default_rule_config

    cfg, model, processor = model_bundle
    suite_name = str(config.get("suite", "libero_spatial"))
    image_size = int(config.get("image_size", 256))
    resize_size = 224
    max_steps = int(args.maximum_steps or config.get("maximum_steps", 220))
    num_steps_wait = int(config.get("num_steps_wait", 10))
    output_dir = ROOT / job["output_path"]
    if args.resume and (output_dir / "COMPLETE").exists():
        ok, _errors, _meta = validate_episode_dir(output_dir, schema)
        if ok:
            return output_dir

    seed = int(args.rollout_seed or job["rollout_seed"])
    set_seed_everywhere(seed)
    env, task, initial_states, bddl_file = build_env_for_task(int(job["task_id"]), suite_name, image_size)
    ontology = read_json(ROOT / "outputs" / "scene_graph_generator_openvla_spatial" / "ontology" / "ontology.json")
    rule_config = default_rule_config(image_size=image_size)
    rule_config["dataset_name"] = "openvla_rollout_graph_v2"
    rule_config["teacher_graph"] = "rule_based_world_graph_live_rollout"

    started = time.time()
    try:
        env.seed(seed)
        obs = env.reset()
        init_id = int(job["initial_state_id"])
        if init_id >= len(initial_states):
            raise IndexError(f"initial_state_id {init_id} >= {len(initial_states)} for task {job['task_id']}")
        obs = env.set_init_state(initial_states[init_id])
        for _ in range(num_steps_wait):
            obs, _reward, _done, _info = env.step(get_libero_dummy_action("openvla"))

        rgbs = []
        y_nodes = []
        y_edges = []
        node_masks = []
        relation_masks = []
        position_targets = []
        position_masks = []
        executed_actions = []
        unnormalized_actions = []
        normalized_actions = []
        frame_records = []
        triplets_by_frame = []
        rewards = []
        dones = []
        success = False
        terminal_reason = "timeout"
        total_reward = 0.0
        peak_gpu_memory = None
        if args.gpu_id >= 0:
            try:
                import torch

                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        t = 0
        while t < max_steps:
            img = get_libero_image(obs, resize_size)
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[-1] != 3:
                raise RuntimeError(f"invalid RGB at t={t}: dtype={img.dtype} shape={img.shape}")
            world_graph, _observable_graph, diagnostics = build_frame_graphs(
                env=env,
                obs=obs,
                bddl_file=bddl_file,
                task_id=str(job["task_id"]),
                demo_id=str(job["job_id"]),
                frame_id=t,
                config=rule_config,
            )
            y_node, y_edge, relation_mask = encode_graph_targets(world_graph, ontology)
            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }
            action_unnorm = get_action(cfg, model, observation, task.language, processor=processor)
            action_exec = normalize_gripper_action(action_unnorm.copy(), binarize=True)
            action_exec = invert_gripper_action(action_exec)
            next_obs, reward, done, info = env.step(action_exec.tolist())

            rgbs.append(img)
            y_nodes.append(y_node)
            y_edges.append(y_edge)
            node_masks.append(y_node.astype(bool))
            relation_masks.append(relation_mask)
            position_targets.append(np.zeros((len(ontology["nodes"]), 3), dtype=np.float32))
            position_masks.append(np.zeros((len(ontology["nodes"]),), dtype=bool))
            executed_actions.append(np.asarray(action_exec, dtype=np.float32))
            unnormalized_actions.append(np.asarray(action_unnorm, dtype=np.float32))
            normalized_actions.append(np.full((7,), np.nan, dtype=np.float32))
            triplets = [
                [edge["subject"], edge["predicate"], edge["object"]]
                for edge in world_graph.get("binary_edges", [])
            ]
            triplets_by_frame.append(triplets)
            contacts = diagnostics.get("contacts", [])
            frame_records.append(
                {
                    "timestep": t,
                    "world_graph_sha256": world_graph.get("metadata", {}).get("graph_sha256"),
                    "triplet_count": len(triplets),
                    "contact_count": len(contacts),
                    "has_grasping": any(edge[1] == "grasping" for edge in triplets),
                    "reward_after_step": float(reward),
                    "done_after_step": bool(done),
                    "info_after_step": dict(info) if isinstance(info, dict) else {},
                }
            )
            rewards.append(float(reward))
            dones.append(bool(done))
            total_reward += float(reward)
            obs = next_obs
            t += 1
            if done:
                success = True
                terminal_reason = "success"
                break

        if args.gpu_id >= 0:
            try:
                import torch

                peak_gpu_memory = int(torch.cuda.max_memory_allocated())
            except Exception:
                peak_gpu_memory = None

        relation_counts = Counter()
        for triplets in triplets_by_frame:
            for _s, pred, _o in triplets:
                relation_counts[pred] += 1
        failure_category = classify_failure(success, terminal_reason, frame_records)
        metadata = {
            "dataset_schema_version": "openvla_rollout_graph_v2",
            "graph_schema_version": schema["graph_schema_version"],
            "suite_name": suite_name,
            "task_id": int(job["task_id"]),
            "task_name": getattr(task, "name", str(job["task_id"])),
            "episode_id": job["job_id"],
            "job_id": job["job_id"],
            "policy_id": args.policy_id,
            "worker_id": args.worker_id,
            "gpu_id": args.gpu_id,
            "initial_state_id": int(job["initial_state_id"]),
            "rollout_seed": seed,
            "raw_instruction": task.language,
            "formatted_prompt": f"In: What action should the robot take to {task.language.lower()}?\nOut:",
            "prompt_template_id": "openvla_default",
            "camera_name": config.get("camera_name", "agentview"),
            "image_size": [224, 224],
            "checkpoint_path": str(policy_runtime.checkpoint),
            "checkpoint_hash": None,
            "action_normalization_key": cfg.unnorm_key,
            "episode_success": success,
            "failure_category": failure_category,
            "failure_category_source": "heuristic" if args.failure_classification == "heuristic" else "none",
            "episode_length": len(rgbs),
            "terminal_reason": terminal_reason,
            "total_reward": total_reward,
            "maximum_steps": max_steps,
            "collection_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "collection_duration_sec": round(time.time() - started, 3),
            "peak_gpu_memory": peak_gpu_memory,
            "depth_used_for_policy": False,
            "depth_used_for_graph_generator_input": False,
            "oracle_uses_simulator_privileged_state": True,
            "relation_counts": dict(relation_counts),
            "frames": frame_records,
            "oracle_graph_triplets": triplets_by_frame,
            "action_normalized_note": "OpenVLA predict_action returns denormalized action; normalized tokens are not exposed by existing eval API.",
        }

        def writer(tmp_dir: Path) -> None:
            np.savez_compressed(
                tmp_dir / "frames.npz",
                rgb=np.stack(rgbs, axis=0),
                oracle_graph_tensor=np.stack(y_edges, axis=0),
                node_valid_mask=np.stack(node_masks, axis=0),
                relation_valid_mask=np.stack(relation_masks, axis=0),
                position_target=np.stack(position_targets, axis=0),
                position_valid_mask=np.stack(position_masks, axis=0),
                predicted_action_normalized=np.stack(normalized_actions, axis=0),
                predicted_action_unnormalized=np.stack(unnormalized_actions, axis=0),
                executed_action=np.stack(executed_actions, axis=0),
                reward=np.asarray(rewards, dtype=np.float32),
                done=np.asarray(dones, dtype=bool),
            )
            write_json(tmp_dir / "metadata.json", metadata)
            (tmp_dir / "COMPLETE").write_text("completed\n", encoding="utf-8")
            checksum_episode_files(tmp_dir)

        atomic_write_episode(output_dir, writer)
    finally:
        try:
            env.close()
        except Exception:
            pass
    if args.validate_after_episode:
        ok, errors, _meta = validate_episode_dir(output_dir, schema)
        if not ok:
            raise RuntimeError(f"validation failed for {output_dir}: {errors}")
    return output_dir


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    config = load_config(args.config)
    policy_pool = load_config(args.policy_config)
    schema = read_json(SCHEMA_LOCK_JSON)
    jobs = get_jobs(args)
    worker_log = args.output_dir / "logs" / f"worker_{args.worker_id:02d}" / "status.jsonl"
    if args.dry_run:
        print(f"dry-run jobs={len(jobs)} policy={args.policy_id} worker={args.worker_id} gpu={args.gpu_id}")
        return
    policy_runtime = resolve_policy_runtime(policy_pool, args.policy_id)
    model_bundle = load_policy(policy_runtime)
    completed = []
    for job in jobs:
        try:
            episode_dir = collect_one_job(job, args, config, policy_runtime, model_bundle, schema)
            row = {"job_id": job["job_id"], "status": "completed", "episode_dir": str(episode_dir), "policy_id": args.policy_id, "worker_id": args.worker_id}
        except Exception as exc:
            row = {"job_id": job["job_id"], "status": "collection_error", "error": repr(exc), "policy_id": args.policy_id, "worker_id": args.worker_id}
            write_jsonl(worker_log, completed + [row])
            raise
        completed.append(row)
        write_jsonl(worker_log, completed)
    print(f"completed {len(completed)} jobs")


if __name__ == "__main__":
    main()
