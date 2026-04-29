from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contact_aware_rl.config import ExperimentConfig, load_experiment_config
from contact_aware_rl.experiment import run_training


DEFAULT_TARGETS = {
    "transport_ready_rate": 0.60,
    "over_goal_rate": 0.50,
    "placement_rate": 0.40,
    "mean_best_goal_distance_xy": 0.16,
}


@dataclass(frozen=True)
class SweepVariant:
    name: str
    seed: int = 0
    env_overrides: dict[str, Any] = field(default_factory=dict)
    reward_overrides: dict[str, Any] = field(default_factory=dict)
    train_overrides: dict[str, Any] = field(default_factory=dict)


DEFAULT_VARIANTS = [
    SweepVariant("ee_nominal_seed0", seed=0),
    SweepVariant("ee_nominal_seed1", seed=1),
    SweepVariant("ee_nominal_seed2", seed=2),
    SweepVariant(
        "ee_wider_step_seed0",
        seed=0,
        env_overrides={
            "action_scale_xyz": 0.025,
            "arm_joint_delta_scales": [0.10, 0.10, 0.10, 0.10],
        },
    ),
    SweepVariant(
        "ee_smoother_seed0",
        seed=0,
        reward_overrides={"action_delta_penalty_weight": 0.001},
    ),
    SweepVariant(
        "ee_long_seed0",
        seed=0,
        train_overrides={"total_timesteps": 1_000_000},
    ),
    SweepVariant(
        "joint_fixed_objective_seed0",
        seed=0,
        env_overrides={"arm_control_mode": "joint_delta"},
    ),
]


def _apply_overrides(config: ExperimentConfig, variant: SweepVariant) -> ExperimentConfig:
    updated = config.clone()
    for key, value in variant.env_overrides.items():
        setattr(updated.env, key, value)
    for key, value in variant.reward_overrides.items():
        setattr(updated.reward, key, value)
    for key, value in variant.train_overrides.items():
        setattr(updated.train, key, value)

    base_tags = list(updated.logging.wandb_tags)
    variant_tags = [
        "arm-closed-loop-sweep",
        f"variant:{variant.name}",
        f"control:{updated.env.arm_control_mode}",
    ]
    updated.logging.wandb_tags = list(dict.fromkeys(base_tags + variant_tags))
    return updated


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _meets_target(metrics: dict[str, Any], targets: dict[str, float]) -> bool:
    if not metrics:
        return False
    return bool(
        float(metrics.get("transport_ready_rate", 0.0))
        >= targets["transport_ready_rate"]
        and float(metrics.get("over_goal_rate", 0.0)) >= targets["over_goal_rate"]
        and float(metrics.get("placement_rate", 0.0)) >= targets["placement_rate"]
        and float(metrics.get("mean_best_goal_distance_xy", float("inf")))
        <= targets["mean_best_goal_distance_xy"]
    )


def _result_row(
    *,
    variant: SweepVariant,
    training_summary: dict[str, Any],
    training_summary_path: Path,
    target_met: bool,
) -> dict[str, Any]:
    best_metrics = training_summary.get("best_validation_metrics") or {}
    test_metrics = training_summary.get("test_metrics") or {}
    return {
        "variant": variant.name,
        "seed": variant.seed,
        "run_id": training_summary.get("run_id"),
        "mode": training_summary.get("mode"),
        "embodiment": training_summary.get("embodiment"),
        "training_status": training_summary.get("training_status"),
        "stop_reason": training_summary.get("stop_reason"),
        "target_met": target_met,
        "best_timestep": training_summary.get("best_timestep"),
        "best_transport_ready_rate": best_metrics.get("transport_ready_rate"),
        "best_over_goal_rate": best_metrics.get("over_goal_rate"),
        "best_placement_rate": best_metrics.get("placement_rate"),
        "best_mean_best_goal_distance_xy": best_metrics.get("mean_best_goal_distance_xy"),
        "best_success_rate": best_metrics.get("success_rate"),
        "test_transport_ready_rate": test_metrics.get("transport_ready_rate"),
        "test_over_goal_rate": test_metrics.get("over_goal_rate"),
        "test_placement_rate": test_metrics.get("placement_rate"),
        "test_mean_best_goal_distance_xy": test_metrics.get("mean_best_goal_distance_xy"),
        "output_dir": training_summary.get("output_dir"),
        "training_summary_path": str(training_summary_path),
    }


def _write_results(
    rows: list[dict[str, Any]],
    output_dir: Path,
    targets: dict[str, float],
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "targets": targets,
        "results": rows,
    }
    (output_dir / "sweep_results.json").write_text(json.dumps(payload, indent=2))

    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "sweep_results.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the closed-loop arm pick-place experiment sweep."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "arm_ee_place_priority.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Sweep ledger directory. Defaults to outputs/arm_closed_loop_sweep/<timestamp>.",
    )
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit zero even if the target is not met. Intended for smoke checks only.",
    )
    parser.add_argument("--target-transport-ready-rate", type=float, default=0.60)
    parser.add_argument("--target-over-goal-rate", type=float, default=0.50)
    parser.add_argument("--target-placement-rate", type=float, default=0.40)
    parser.add_argument("--target-mean-best-goal-distance-xy", type=float, default=0.16)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    base_config = load_experiment_config(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "arm_closed_loop_sweep" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "transport_ready_rate": args.target_transport_ready_rate,
        "over_goal_rate": args.target_over_goal_rate,
        "placement_rate": args.target_placement_rate,
        "mean_best_goal_distance_xy": args.target_mean_best_goal_distance_xy,
    }
    variants = DEFAULT_VARIANTS[: args.max_runs] if args.max_runs else DEFAULT_VARIANTS
    rows: list[dict[str, Any]] = []
    target_met = False
    for variant in variants:
        config = _apply_overrides(base_config, variant)
        config.logging.output_root = str(output_dir / "runs" / variant.name)
        if args.num_envs is not None:
            config.train.num_envs = args.num_envs
        if args.total_timesteps is not None:
            config.train.total_timesteps = args.total_timesteps

        artifacts = run_training(
            config,
            mode="contact",
            seed=variant.seed,
            num_envs=config.train.num_envs,
            total_timesteps=config.train.total_timesteps,
            output_root=config.logging.output_root,
            wandb_mode=args.wandb_mode,
        )
        training_summary = _load_json(artifacts.training_summary_path)
        best_metrics = training_summary.get("best_validation_metrics") or {}
        target_met = _meets_target(best_metrics, targets)
        rows.append(
            _result_row(
                variant=variant,
                training_summary=training_summary,
                training_summary_path=artifacts.training_summary_path,
                target_met=target_met,
            )
        )
        _write_results(rows, output_dir, targets)

        if target_met:
            break

    print(json.dumps({"output_dir": str(output_dir), "target_met": target_met, "results": rows}, indent=2))
    return 0 if target_met or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
