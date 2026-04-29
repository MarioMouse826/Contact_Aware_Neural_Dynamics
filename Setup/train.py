"""
Train SAC on HumanoidLiftEnv with parallel envs and W&B logging.

Usage:
    python Setup/train.py --steps 200000 --num-envs 8

Env vars:
    ENV_USE_CONTACT_BITS=1  # default; 0 = ablation baseline
    WANDB_MODE=offline      # if you want to skip W&B sync
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Spawn start method - required on macOS for SubprocVecEnv with MuJoCo
if sys.platform == "darwin":
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

import numpy as np
import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from wandb.integration.sb3 import WandbCallback

from utils import load_wandb_config, resolve_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200_000,
                   help="Total training timesteps")
    p.add_argument("--num-envs", type=int, default=8,
                   help="Number of parallel envs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--no-contact", action="store_true",
                   help="Ablation: disable contact bits")
    p.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc",
                   help="subproc = true parallel; dummy = sequential (debugging)")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to .zip model to resume from")
    return p.parse_args()


def make_env(seed: int, use_contact_bits: bool):
    def _init():
        from humanoid_lift_env import HumanoidLiftEnv
        env = HumanoidLiftEnv(use_contact_bits=use_contact_bits)
        env.reset(seed=seed)
        return env
    return _init


class EarlyStopOnFallCallback(BaseCallback):
    """Terminate training if fall rate stays at 1.0 over a sustained window.

    Conditions to trigger:
      - At least `min_steps` of training have elapsed (let SAC bootstrap).
      - At least `window` episodes have completed.
      - All of the last `window` episodes ended in a fall.
    """

    def __init__(self, window: int = 100, min_steps: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.window = window
        self.min_steps = min_steps
        self.fall_history: list[int] = []

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            info = self.locals["infos"][i]
            self.fall_history.append(int(info.get("fell", False)))

        # Need enough warmup and enough episodes
        if self.num_timesteps < self.min_steps:
            return True
        if len(self.fall_history) < self.window:
            return True

        recent = self.fall_history[-self.window:]
        if sum(recent) == self.window:
            print(f"\n[EarlyStop] Fall rate = 1.0 over last {self.window} episodes "
                  f"at step {self.num_timesteps}. Terminating training.")
            self.logger.record("custom/early_stopped", 1.0)
            return False  # signal SB3 to stop training

        return True


class CustomMetricsCallback(BaseCallback):
    """Log task-specific metrics on episode end."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.successes = []
        self.falls = []
        self.final_dists = []
        self.final_block_zs = []

    def _on_step(self) -> bool:
        # Vec env: dones is array of bools
        for i, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            info = self.locals["infos"][i]
            self.successes.append(int(info.get("success", False)))
            self.falls.append(int(info.get("fell", False)))
            self.final_dists.append(info.get("dist_to_block", float("nan")))
            self.final_block_zs.append(info.get("block_z", float("nan")))

            # Log running averages every 50 episodes
            if len(self.successes) >= 50:
                self.logger.record("custom/success_rate",
                                   float(np.mean(self.successes[-50:])))
                self.logger.record("custom/fall_rate",
                                   float(np.mean(self.falls[-50:])))
                self.logger.record("custom/final_dist_mean",
                                   float(np.mean(self.final_dists[-50:])))
                self.logger.record("custom/final_block_z_mean",
                                   float(np.mean(self.final_block_zs[-50:])))
        return True


def main():
    args = parse_args()

    # Apply contact ablation via env var (so child processes inherit it)
    if args.no_contact:
        os.environ["ENV_USE_CONTACT_BITS"] = "0"
        mode_str = "no-contact"
    else:
        os.environ["ENV_USE_CONTACT_BITS"] = "1"
        mode_str = "contact"

    # Config
    cfg = load_wandb_config()
    device = resolve_device(cfg.get("device", "auto"))
    print(f"Device: {device}")

    # W&B init
    run_name = args.run_name or f"g1-lift-{mode_str}-{int(time.time())}"
    run = wandb.init(
        project=cfg.get("project", "contact-aware-rl"),
        entity=cfg.get("entity"),
        group=cfg.get("group", "humanoid-lift"),
        tags=cfg.get("tags", []) + [mode_str],
        name=run_name,
        sync_tensorboard=True,
        config={
            "algorithm": "SAC",
            "total_timesteps": args.steps,
            "num_envs": args.num_envs,
            "use_contact_bits": not args.no_contact,
            "seed": args.seed,
            "device": device,
        },
    )

    # Vectorized env
    print(f"Building {args.num_envs} parallel envs ({args.vec_env})...")
    env_fns = [make_env(args.seed + i, not args.no_contact)
               for i in range(args.num_envs)]
    if args.vec_env == "subproc" and args.num_envs > 1:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # SAC model — load existing or create new
    if args.resume_from:
        ckpt = Path(args.resume_from)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print(f"Resuming from {ckpt}")
        model = SAC.load(
            str(ckpt),
            env=vec_env,
            device=device,
            tensorboard_log=f"runs/{run.id}",
        )
        # Load replay buffer if present
        buffer_path = ckpt.with_name(ckpt.stem + "_buffer.pkl")
        if buffer_path.exists():
            print(f"Loading replay buffer from {buffer_path}")
            model.load_replay_buffer(str(buffer_path))
        else:
            print(f"No replay buffer at {buffer_path}; starting with fresh buffer "
                  f"(agent will need to refill before learning resumes).")
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_update_interval=1,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=f"runs/{run.id}",
            device=device,
            seed=args.seed,
            verbose=1,
        )

    # Callbacks
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.num_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix="sac_g1_lift",
        ),
        CustomMetricsCallback(),
        EarlyStopOnFallCallback(window=100, min_steps=10_000),
        WandbCallback(
            gradient_save_freq=10_000,
            verbose=2,
        ),
    ]

    # Train
    print(f"\nStarting training for {args.steps:,} steps...")
    t0 = time.time()
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/60:.1f} min ({args.steps/elapsed:.0f} steps/sec)")

    # Save final model + replay buffer at repo root
    final_path = REPO_ROOT / "sac_humanoid_lifter"
    model.save(str(final_path))
    buffer_path = REPO_ROOT / "sac_humanoid_lifter_buffer.pkl"
    model.save_replay_buffer(str(buffer_path))
    print(f"Saved final model to {final_path}.zip")
    print(f"Saved replay buffer to {buffer_path}")

    vec_env.close()
    wandb.finish()


if __name__ == "__main__":
    main()