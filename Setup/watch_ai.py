from __future__ import annotations

import argparse
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = "sac_humanoid_lifter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained SAC policy in the MuJoCo viewer.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sleep-seconds", type=float, default=0.025)
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def main() -> None:
    args = parse_args()

    try:
        import mujoco
        import mujoco.viewer
        from stable_baselines3 import SAC
        from humanoid_env import HumanoidLifterEnv
    except ImportError as exc:
        raise SystemExit(
            "Viewer dependencies are missing. Install the project dependencies with `pip install -e .` or `uv sync`."
        ) from exc

    model_path = resolve_repo_path(args.model_path)

    print("Loading environment...")
    env = HumanoidLifterEnv()

    print(f"Loading model from {model_path}...")
    model = SAC.load(str(model_path))

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
