from __future__ import annotations

from .runtime import PROJECT_ROOT, configure_runtime_environment

configure_runtime_environment()

import argparse
import json

from .evaluation import DEFAULT_EVAL_SPLIT, EVAL_SPLITS
from .experiment import evaluate_checkpoint
from .modes import EVALUATION_MODES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved SAC checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to the .zip checkpoint.")
    parser.add_argument("--mode", choices=sorted(EVALUATION_MODES), required=True)
    parser.add_argument("--config", default=None, help="Optional config path override.")
    parser.add_argument(
        "--split",
        choices=EVAL_SPLITS,
        default=DEFAULT_EVAL_SPLIT,
        help="Named evaluation split. Use `custom` together with --base-seed.",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument(
        "--default-config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = evaluate_checkpoint(
        args.checkpoint,
        mode=args.mode,
        config_path=args.config,
        split=args.split,
        episodes=args.episodes,
        base_seed=args.base_seed,
        output_path=args.output_path,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
