from __future__ import annotations

from .runtime import PROJECT_ROOT, configure_runtime_environment

configure_runtime_environment()

import argparse
import json

from .config import load_experiment_config
from .experiment import run_proposal_suite


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the proposal experiment suite.")
    parser.add_argument("--suite", choices=["proposal"], required=True)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to a YAML config file.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "proposal-suite"),
        help="Where to write the aggregated suite results.",
    )
    parser.add_argument("--wandb-mode", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_experiment_config(args.config)
    payload = run_proposal_suite(
        config,
        seeds=args.seeds,
        num_envs=args.num_envs,
        output_dir=args.output_dir,
        wandb_mode=args.wandb_mode,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
