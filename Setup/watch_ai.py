from __future__ import annotations
import argparse
import time
from pathlib import Path

from utils import load_wandb_config, resolve_device

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = "sac_humanoid_lifter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained SAC policy in MuJoCo, optionally saving an mp4.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sleep-seconds", type=float, default=0.025)
    parser.add_argument("--record", action="store_true", help="Record an mp4 instead of opening live viewer.")
    parser.add_argument("--output", default="humanoid_run.mp4", help="Output mp4 path (with --record).")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record (with --record).")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def main() -> None:
    args = parse_args()
    wandb_config = load_wandb_config()
    requested_device = wandb_config["device"]
    resolved_device = resolve_device(requested_device)

    try:
        import mujoco
        import mujoco.viewer
        from stable_baselines3 import SAC
        from humanoid_env import HumanoidLifterEnv
    except ImportError as exc:
        raise SystemExit(
            "Viewer dependencies are missing. Install with `pip install -e .` or `uv sync`."
        ) from exc

    model_path = resolve_repo_path(args.model_path)
    print("Loading environment...")
    env = HumanoidLifterEnv()
    print(f"Loading model from {model_path}...")
    print(f"Using device: {resolved_device} (requested: {requested_device})")
    model = SAC.load(str(model_path), device=resolved_device)

    if args.record:
        import imageio.v2 as imageio
        print(f"Recording {args.episodes} episodes to {args.output} at {args.width}x{args.height} @ {args.fps}fps...")
        renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
        frames = []
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                renderer.update_scene(env.data)
                frames.append(renderer.render())
                steps += 1
                done = terminated or truncated
            print(f"  Episode {ep + 1} ended after {steps} steps.")
        imageio.mimsave(args.output, frames, fps=args.fps, codec="libx264")
        print(f"Saved {len(frames)} frames to {args.output}.")
        return

    obs, _ = env.reset()
    print("Opening viewer...")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(args.sleep_seconds)
            if terminated or truncated:
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
