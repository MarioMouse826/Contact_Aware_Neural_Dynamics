"""
Watch a trained SAC humanoid policy.

Modes:
    Live viewer (default):
        mjpython Setup/watch_ai.py
    Record mp4:
        python Setup/watch_ai.py --record --output humanoid_run.mp4
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = "sac_humanoid_lifter"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--sleep-seconds", type=float, default=0.025)
    p.add_argument("--record", action="store_true",
                   help="Record an mp4 instead of opening live viewer")
    p.add_argument("--output", default="humanoid_run.mp4")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def main():
    args = parse_args()

    from utils import load_wandb_config, resolve_device
    cfg = load_wandb_config()
    device = resolve_device(cfg.get("device", "auto"))

    import mujoco
    from stable_baselines3 import SAC
    from humanoid_lift_env import HumanoidLiftEnv

    model_path = resolve_repo_path(args.model_path)
    print(f"Loading env...")
    env = HumanoidLiftEnv()
    print(f"Loading model from {model_path} (device={device})...")
    model = SAC.load(str(model_path), device=device)

    if args.record:
        import imageio.v2 as imageio
        print(f"Recording {args.episodes} episodes to {args.output} "
              f"at {args.width}x{args.height} @ {args.fps}fps...")
        renderer = mujoco.Renderer(env.model,
                                    height=args.height, width=args.width)
        frames = []
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                renderer.update_scene(env.data)
                frames.append(renderer.render())
                steps += 1
                done = terminated or truncated
            print(f"  Episode {ep+1}: {steps} steps, "
                  f"final dist={info['dist_to_block']:.2f}m, "
                  f"final block_z={info['block_z']:.3f}m, "
                  f"success={info.get('success', False)}")
        imageio.mimsave(args.output, frames, fps=args.fps, codec="libx264")
        print(f"Saved {len(frames)} frames to {args.output}.")
        env.close()
        return

    # Live viewer
    import mujoco.viewer
    obs, _ = env.reset()
    print("Opening viewer (close window to exit)...")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(args.sleep_seconds)
            if terminated or truncated:
                obs, _ = env.reset()


if __name__ == "__main__":
    main()