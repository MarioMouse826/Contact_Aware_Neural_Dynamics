from __future__ import annotations

from .runtime import PROJECT_ROOT, configure_runtime_environment

configure_runtime_environment()

import argparse
import json

from .config import load_experiment_config
from .experiment import run_training
from .modes import TRAINABLE_MODES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SAC on the configured manipulation task."
    )
    parser.add_argument("--mode", choices=sorted(TRAINABLE_MODES), required=True)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to a YAML config file.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_experiment_config(args.config)
    artifacts = run_training(
        config,
        mode=args.mode,
        seed=args.seed,
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        output_root=args.output_root,
        wandb_mode=args.wandb_mode,
        init_checkpoint=args.init_checkpoint,
    )
    print(
        json.dumps(
            {
                "run_id": artifacts.run_id,
                "mode": artifacts.mode,
                "output_dir": str(artifacts.output_dir),
                "best_model_path": str(artifacts.best_model_path),
                "best_success_model_path": str(artifacts.best_success_model_path),
                "latest_model_path": str(artifacts.latest_model_path),
                "final_model_path": str(artifacts.final_model_path),
                "training_summary_path": str(artifacts.training_summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
