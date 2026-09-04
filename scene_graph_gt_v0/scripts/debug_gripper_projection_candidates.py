#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from scene_graph.rule_generator import (  # noqa: E402
    create_env,
    h5_attr_text,
    resolve_task,
    set_demo_state,
    sorted_demo_keys,
)
from scene_graph.node_extractor import (  # noqa: E402
    camera_record,
    model_name_to_id,
    named_world_position,
    project_world_to_image,
    site_world_position,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scene_graph_gt_v0"))
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--language-substring", default="akita black bowl")
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+", default=[49, 78, 97])
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    import h5py
    import numpy as np
    from PIL import Image, ImageDraw

    args = parse_args()
    task = resolve_task(args.suite, args.language_substring)
    env = create_env(task["bddl_file"], args.image_size)
    out_dir = args.output_dir / "debug_gripper_projection_candidates" / "demo_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "site": [
            "gripper0_ft_frame",
            "gripper0_grip_site",
            "gripper0_ee_x",
            "gripper0_ee_y",
            "gripper0_ee_z",
            "gripper0_grip_site_cylinder",
        ],
        "geom": [
            "gripper0_hand_visual",
            "gripper0_hand_collision",
            "gripper0_finger1_visual",
            "gripper0_finger1_collision",
            "gripper0_finger1_pad_collision",
            "gripper0_finger2_visual",
            "gripper0_finger2_collision",
            "gripper0_finger2_pad_collision",
        ],
    }
    try:
        with h5py.File(task["demo_file"], "r") as f:
            demo_key = sorted_demo_keys(f["data"])[args.demo_index]
            group = f[f"data/{demo_key}"]
            states = group["states"][()]
            model_xml = h5_attr_text(group.attrs.get("model_file"))
            loaded_xml = False
            for frame_id in args.frames:
                obs = set_demo_state(env, states[frame_id], model_xml if not loaded_xml else None)
                loaded_xml = True
                model = env.sim.model
                data = env.sim.data
                camera = camera_record(model, data, "agentview")
                image = Image.fromarray(np.asarray(obs["agentview_image"])).convert("RGB").transpose(Image.Transpose.ROTATE_180)
                draw = ImageDraw.Draw(image, "RGBA")
                rows = []
                seg = np.asarray(obs["agentview_segmentation_instance"])
                if seg.ndim == 3:
                    seg = seg[..., 0]
                robot_seg_ids = []
                robot_id = getattr(env, "segmentation_robot_id", None)
                if robot_id is not None:
                    robot_seg_ids.append(int(robot_id) + 1)
                for instance_name, seg_id in dict(getattr(env, "instance_to_id", {}) or {}).items():
                    lowered = str(instance_name).lower()
                    if any(token in lowered for token in ("panda", "gripper", "robot")):
                        robot_seg_ids.append(int(seg_id))
                if robot_seg_ids:
                    mask = np.isin(seg, sorted(set(robot_seg_ids)))
                    ys, xs = np.where(mask)
                    if xs.size:
                        y_cut = float(np.quantile(ys, 0.20))
                        keep = ys <= y_cut
                        if int(keep.sum()) < 8:
                            y_cut = float(np.quantile(ys, 0.35))
                            keep = ys <= y_cut
                        mx = float(xs[keep].mean())
                        my = float(ys[keep].mean())
                        rx = args.image_size - 1 - mx
                        ry = args.image_size - 1 - my
                        draw.ellipse((rx - 7, ry - 7, rx + 7, ry + 7), fill=(170, 40, 220, 230), outline=(255, 255, 255, 255), width=2)
                        draw.text((rx + 8, ry - 7), "M", fill=(0, 0, 0, 255))
                        rows.append(
                            {
                                "idx": "M",
                                "kind": "robot_mask_frontier",
                                "name": "robot_mask_upper_20pct",
                                "projected_xy": [mx, my],
                                "rotated_xy": [rx, ry],
                                "visible_pixels": int(xs.size),
                                "frontier_pixels": int(keep.sum()),
                            }
                        )
                idx = 1
                for kind, kind_names in names.items():
                    for name in kind_names:
                        if model_name_to_id(model, kind, name) is None:
                            continue
                        point = site_world_position(model, data, name) if kind == "site" else named_world_position(model, data, kind, name)
                        projected = project_world_to_image(point, camera, args.image_size, args.image_size) if point is not None and camera else None
                        if projected is None:
                            rows.append({"idx": idx, "kind": kind, "name": name, "projected_xy": None})
                            idx += 1
                            continue
                        x = args.image_size - 1 - float(projected[0])
                        y = args.image_size - 1 - float(projected[1])
                        color = (255, 40, 40, 230) if kind == "site" else (30, 100, 255, 230)
                        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(255, 255, 255, 255), width=1)
                        draw.text((x + 6, y - 6), str(idx), fill=(0, 0, 0, 255))
                        rows.append({"idx": idx, "kind": kind, "name": name, "projected_xy": [float(projected[0]), float(projected[1])], "rotated_xy": [x, y]})
                        idx += 1
                draw.rectangle((0, 0, image.width, 18), fill=(255, 255, 255, 220))
                draw.text((4, 3), f"frame {frame_id}: red=site blue=geom", fill=(0, 0, 0, 255))
                image.save(out_dir / f"{frame_id:06d}.png")
                (out_dir / f"{frame_id:06d}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    finally:
        env.close()
    print(json.dumps({"debug_dir": str(out_dir), "frames": args.frames}, indent=2))


if __name__ == "__main__":
    main()
