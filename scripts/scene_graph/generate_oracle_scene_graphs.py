#!/usr/bin/env python3
"""Generate oracle Scene Graph / HOI Graph JSONL from LIBERO simulator state."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

from oracle_scene_graph_utils import (
    GraphThresholds,
    append_jsonl,
    create_libero_env,
    exception_payload,
    make_graph_record,
    reset_env_to_episode,
    safe_json_dump,
    save_rgb_from_obs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_probe"))
    parser.add_argument("--policy", choices=["zero", "random"], default="zero")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-rgb", action="store_true")
    parser.add_argument("--next-to-threshold", type=float, default=0.12)
    parser.add_argument("--between-distance-threshold", type=float, default=0.06)
    parser.add_argument("--between-min-endpoint-distance", type=float, default=0.04)
    parser.add_argument("--between-min-pair-distance", type=float, default=0.08)
    parser.add_argument("--on-xy-threshold", type=float, default=0.09)
    parser.add_argument("--on-z-threshold", type=float, default=0.015)
    parser.add_argument("--inside-xy-threshold", type=float, default=0.10)
    parser.add_argument("--inside-z-threshold", type=float, default=0.12)
    parser.add_argument("--grasp-distance-threshold", type=float, default=0.08)
    return parser.parse_args()


def sample_action(policy: str) -> List[float]:
    if policy == "random":
        return [random.uniform(-0.05, 0.05) for _ in range(6)] + [random.choice([-1.0, 1.0])]
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def write_generation_readme(output_dir: Path, args: argparse.Namespace) -> None:
    readme = output_dir / "README_oracle_scene_graphs.md"
    readme.write_text(
        "\n".join(
            [
                "# Oracle LIBERO Scene Graph Generation",
                "",
                "This probe generates Paper 1 oracle graphs from LIBERO / robosuite / MuJoCo simulator state only. No external perception model is used.",
                "",
                "## Re-run",
                "",
                "```bash",
                "python scripts/scene_graph/generate_oracle_scene_graphs.py \\",
                f"  --suite {args.suite} --task-id {args.task_id} \\",
                f"  --num-episodes {args.num_episodes} --max-steps {args.max_steps} --sample-every {args.sample_every} \\",
                f"  --output-dir {args.output_dir} --policy {args.policy} --save-rgb",
                "```",
                "",
                "## Limits",
                "",
                "- Default rollout policy is zero-action unless `--policy random` is selected.",
                "- `on` and `inside` are approximate center/contact rules until geometry extents are integrated.",
                "- `grasping` is a candidate relation based on gripper-object contact plus distance.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = args.output_dir / "graphs"
    rgb_dir = args.output_dir / "rgb"
    summary = {"episodes": [], "errors": [], "config": vars(args)}

    thresholds = GraphThresholds(
        next_to=args.next_to_threshold,
        between_distance=args.between_distance_threshold,
        between_min_endpoint_distance=args.between_min_endpoint_distance,
        between_min_pair_distance=args.between_min_pair_distance,
        on_xy=args.on_xy_threshold,
        on_z=args.on_z_threshold,
        inside_xy=args.inside_xy_threshold,
        inside_z=args.inside_z_threshold,
        grasp_distance=args.grasp_distance_threshold,
    )

    env = None
    try:
        env, _task_suite, _task, init_states, metadata = create_libero_env(
            args.suite,
            args.task_id,
            image_size=args.image_size,
            camera_depths=False,
            camera_segmentations=None,
            seed=args.seed,
        )
        safe_json_dump(metadata, args.output_dir / "generation_task_metadata.json")
        for episode_id in range(args.num_episodes):
            graph_path = graphs_dir / f"episode_{episode_id:06d}.jsonl"
            if graph_path.exists():
                graph_path.unlink()
            episode_summary = {"episode_id": episode_id, "records": 0, "error": None}
            try:
                obs, warnings = reset_env_to_episode(env, init_states, episode_id)
                for timestep in range(args.max_steps + 1):
                    if timestep % args.sample_every == 0:
                        record = make_graph_record(
                            suite=args.suite,
                            task_id=args.task_id,
                            task_name=metadata.get("task_name", str(args.task_id)),
                            instruction=metadata.get("instruction", ""),
                            episode_id=episode_id,
                            timestep=timestep,
                            env=env,
                            thresholds=thresholds,
                            warnings=list(warnings),
                            image_width=args.image_size,
                            image_height=args.image_size,
                        )
                        if args.save_rgb:
                            image_path = rgb_dir / f"episode_{episode_id:06d}" / f"t{timestep:06d}.png"
                            saved = save_rgb_from_obs(obs, image_path)
                            record["rgb_path"] = str(saved) if saved else None
                        append_jsonl(graph_path, record)
                        episode_summary["records"] += 1
                    if timestep >= args.max_steps:
                        break
                    obs, _reward, done, _info = env.step(sample_action(args.policy))
                    if done:
                        break
            except Exception as exc:
                episode_summary["error"] = exception_payload(exc)
                summary["errors"].append({"episode_id": episode_id, "error": episode_summary["error"]})
            summary["episodes"].append(episode_summary)
        write_generation_readme(args.output_dir, args)
        safe_json_dump(summary, args.output_dir / "graph_generation_run_summary.json")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
