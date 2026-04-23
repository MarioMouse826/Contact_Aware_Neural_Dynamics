#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
from stable_baselines3 import SAC

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contact_aware_rl.config import load_experiment_config
from contact_aware_rl.env import BaseContactAwareEnv, make_env
from contact_aware_rl.evaluation import DEFAULT_EVAL_SPLIT, EVAL_SPLITS, resolve_eval_split
from contact_aware_rl.modes import apply_mode_overrides, infer_mode_from_env_config, resolve_mode
from contact_aware_rl.runtime import default_video_stem


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a saved SAC checkpoint to an MP4.")
    parser.add_argument("--model-path", required=True, help="Path to a saved `.zip` SAC model.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "contact", "always_contact", "contact_ablation"],
        default=None,
        help="Evaluation mode override. Defaults to the checkpoint training mode.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config override. Defaults to `<checkpoint_dir>/config.yaml`.",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Output MP4 path. Defaults to `videos/<checkpoint-dir>.mp4`.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record.")
    parser.add_argument(
        "--split",
        choices=EVAL_SPLITS,
        default=DEFAULT_EVAL_SPLIT,
        help="Named evaluation split. Use `custom` together with --base-seed.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Override the base seed for `custom` playback.",
    )
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on steps per episode.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Video FPS. Defaults to the environment control rate.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Video width in pixels. Defaults to 960 if the framebuffer allows it.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Video height in pixels. Defaults to 720 if the framebuffer allows it.",
    )
    parser.add_argument("--camera", default="overview", help="MuJoCo camera name to render.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy actions.",
    )
    return parser


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _resolve_checkpoint_path(model_path_arg: str) -> Path:
    model_path = Path(model_path_arg).expanduser()
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()
    else:
        model_path = model_path.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    return model_path


def _resolve_mode_for_checkpoint(
    *,
    checkpoint_path: Path,
    config_path_arg: str | None,
    requested_mode: str | None,
) -> tuple[str, Path]:
    config_path = (
        Path(config_path_arg).expanduser().resolve()
        if config_path_arg is not None
        else checkpoint_path.parent / "config.yaml"
    )
    config = load_experiment_config(config_path)
    metadata = _load_json_if_exists(checkpoint_path.parent / "metadata.json")
    training_mode = metadata.get("mode", infer_mode_from_env_config(config))

    mode = requested_mode or training_mode
    resolve_mode(mode)

    valid_modes = {training_mode}
    if training_mode == "contact":
        valid_modes.add("contact_ablation")
    if mode not in valid_modes:
        raise ValueError(
            f"Checkpoint trained in mode '{training_mode}' cannot be replayed with mode '{mode}'."
        )
    return mode, config_path


def _default_output_video(model_path_arg: str) -> Path:
    target = PROJECT_ROOT / "videos" / f"{default_video_stem(model_path_arg)}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _default_fps(env: BaseContactAwareEnv) -> int:
    seconds_per_action = float(env.model.opt.timestep) * int(env.env_config.substeps)
    if seconds_per_action <= 0:
        return 20
    return max(1, int(round(1.0 / seconds_per_action)))


def _capture_frame(
    renderer: mujoco.Renderer,
    env: BaseContactAwareEnv,
    *,
    camera: str,
) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera)
    return np.ascontiguousarray(renderer.render())


def _open_writer(output_video: Path, *, width: int, height: int, fps: int) -> cv2.VideoWriter:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_video}")
    return writer


def _resolve_render_size(
    env: BaseContactAwareEnv,
    *,
    requested_width: int | None,
    requested_height: int | None,
) -> tuple[int, int]:
    max_width = int(env.model.vis.global_.offwidth)
    max_height = int(env.model.vis.global_.offheight)
    width = requested_width if requested_width is not None else min(960, max_width)
    height = requested_height if requested_height is not None else min(720, max_height)

    if width < 1 or height < 1:
        raise ValueError(f"Render size must be positive, got {width}x{height}.")
    if width > max_width or height > max_height:
        raise ValueError(
            "Requested render size "
            f"{width}x{height} exceeds the MuJoCo offscreen framebuffer "
            f"{max_width}x{max_height}. Reduce --width/--height or increase the "
            "model's <visual><global offwidth=... offheight=.../> settings."
        )
    return width, height


def record_policy_video(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = _resolve_checkpoint_path(args.model_path)
    mode, config_path = _resolve_mode_for_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path_arg=args.config,
        requested_mode=args.mode,
    )

    config = load_experiment_config(config_path)
    eval_config = apply_mode_overrides(config, mode)
    output_video = (
        Path(args.output_video).expanduser().resolve()
        if args.output_video is not None
        else _default_output_video(checkpoint_path)
    )

    env = make_env(eval_config.env, eval_config.reward)
    renderer = None
    writer = None
    try:
        model = SAC.load(str(checkpoint_path), env=env, device=eval_config.train.device)
        fps = args.fps or _default_fps(env)
        width, height = _resolve_render_size(
            env,
            requested_width=args.width,
            requested_height=args.height,
        )
        writer = _open_writer(output_video, width=width, height=height, fps=fps)
        renderer = mujoco.Renderer(env.model, height=height, width=width)

        seed_override = args.base_seed if args.base_seed is not None else args.seed
        resolved_split, _, base_seed = resolve_eval_split(
            eval_config,
            split=args.split,
            episodes=args.episodes,
            base_seed=seed_override,
        )
        max_steps = args.max_steps or int(eval_config.env.max_episode_steps)

        episodes: list[dict[str, Any]] = []
        for episode_index in range(args.episodes):
            observation, info = env.reset(seed=base_seed + episode_index)
            writer.write(
                cv2.cvtColor(
                    _capture_frame(renderer, env, camera=args.camera),
                    cv2.COLOR_RGB2BGR,
                )
            )

            terminated = False
            truncated = False
            episode_return = 0.0
            steps = 0

            while not (terminated or truncated) and steps < max_steps:
                action, _state = model.predict(observation, deterministic=args.deterministic)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                steps += 1
                writer.write(
                    cv2.cvtColor(
                        _capture_frame(renderer, env, camera=args.camera),
                        cv2.COLOR_RGB2BGR,
                    )
                )

            episodes.append(
                {
                    "episode": episode_index,
                    "seed": base_seed + episode_index,
                    "task": info.get("task"),
                    "return": episode_return,
                    "steps": steps,
                    "success": bool(info.get("is_success", 0.0) >= 0.5),
                    "termination_reason": info.get("termination_reason"),
                    "max_lift_height": float(info.get("max_lift_height", 0.0)),
                    "goal_distance_xy": float(info.get("goal_distance_xy", 0.0)),
                    "is_placed": bool(info.get("is_placed", 0.0) >= 0.5),
                    "is_released": bool(info.get("is_released", 0.0) >= 0.5),
                    "is_settled": bool(info.get("is_settled", 0.0) >= 0.5),
                    "contact_stability": float(info.get("contact_stability", 0.0)),
                }
            )

        return {
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
            "mode": mode,
            "split": resolved_split,
            "base_seed": base_seed,
            "output_video": str(output_video),
            "episodes": episodes,
            "fps": fps,
            "width": width,
            "height": height,
        }
    finally:
        if renderer is not None:
            renderer.close()
        if writer is not None:
            writer.release()
        env.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = record_policy_video(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
