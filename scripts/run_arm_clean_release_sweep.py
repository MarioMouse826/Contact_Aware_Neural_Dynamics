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
from contact_aware_rl.experiment import TrainingArtifacts, run_training


@dataclass(frozen=True)
class SweepRecipe:
    name: str
    seed: int = 0
    env_overrides: dict[str, Any] = field(default_factory=dict)
    reward_overrides: dict[str, Any] = field(default_factory=dict)
    train_overrides: dict[str, Any] = field(default_factory=dict)


DEFAULT_RECIPES = [
    SweepRecipe("formula_nominal", seed=0),
    SweepRecipe("formula_seed1", seed=1),
    SweepRecipe(
        "carry_soft",
        seed=0,
        reward_overrides={
            "carry_height_bonus_weight": 0.35,
            "transport_vertical_speed_penalty_weight": 0.35,
        },
    ),
    SweepRecipe(
        "release_plus",
        seed=0,
        reward_overrides={
            "release_weight": 2.5,
            "release_ready_open_bonus_weight": 0.25,
            "release_ready_hold_penalty_weight": 0.08,
            "post_release_retreat_bonus_weight": 0.18,
            "post_release_recontact_penalty_weight": 0.22,
        },
    ),
    SweepRecipe(
        "low_lr_smooth",
        seed=0,
        env_overrides={"action_smoothing": 0.08},
        train_overrides={"learning_rate": 1.5e-4},
    ),
]


def _dedupe_tags(tags: list[str]) -> list[str]:
    return list(dict.fromkeys(tags))


def _apply_recipe(
    config: ExperimentConfig,
    recipe: SweepRecipe,
    *,
    stage: str,
    output_root: Path,
) -> ExperimentConfig:
    updated = config.clone()
    for key, value in recipe.env_overrides.items():
        setattr(updated.env, key, value)
    for key, value in recipe.reward_overrides.items():
        setattr(updated.reward, key, value)
    for key, value in recipe.train_overrides.items():
        setattr(updated.train, key, value)

    updated.train.seed = recipe.seed
    updated.logging.output_root = str(output_root)
    updated.logging.wandb_tags = _dedupe_tags(
        [
            *updated.logging.wandb_tags,
            "arm-clean-release-sweep",
            "source:iconic-haze-72",
            f"recipe:{recipe.name}",
            f"stage:{stage}",
            f"control:{updated.env.arm_control_mode}",
        ]
    )
    return updated


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_required_checkpoint(path_like: str) -> str:
    checkpoint = Path(path_like).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Initial checkpoint does not exist: {checkpoint}")
    return str(checkpoint)


def _success_rate(metrics: dict[str, Any] | None) -> float:
    return float((metrics or {}).get("success_rate", 0.0))


def _meets_success_target(
    metrics: dict[str, Any] | None,
    *,
    target_success_rate: float,
) -> bool:
    return _success_rate(metrics) >= target_success_rate


def _should_stop_after_stage(*, target_met: bool, stop_on_target: bool) -> bool:
    return target_met and stop_on_target


def _result_row(
    *,
    recipe: SweepRecipe,
    stage: str,
    init_checkpoint: str,
    training_summary: dict[str, Any],
    training_summary_path: Path,
    target_met: bool,
) -> dict[str, Any]:
    best_metrics = training_summary.get("best_validation_metrics") or {}
    test_metrics = training_summary.get("test_metrics") or {}
    return {
        "recipe": recipe.name,
        "stage": stage,
        "seed": recipe.seed,
        "run_id": training_summary.get("run_id"),
        "mode": training_summary.get("mode"),
        "embodiment": training_summary.get("embodiment"),
        "init_checkpoint_path": init_checkpoint,
        "training_status": training_summary.get("training_status"),
        "stop_reason": training_summary.get("stop_reason"),
        "target_met": target_met,
        "best_timestep": training_summary.get("best_timestep"),
        "best_success_rate": best_metrics.get("success_rate"),
        "best_transport_ready_rate": best_metrics.get("transport_ready_rate"),
        "best_over_goal_rate": best_metrics.get("over_goal_rate"),
        "best_placement_rate": best_metrics.get("placement_rate"),
        "best_release_rate": best_metrics.get("release_rate"),
        "best_mean_best_goal_distance_xy": best_metrics.get("mean_best_goal_distance_xy"),
        "test_success_rate": test_metrics.get("success_rate"),
        "test_transport_ready_rate": test_metrics.get("transport_ready_rate"),
        "test_over_goal_rate": test_metrics.get("over_goal_rate"),
        "test_placement_rate": test_metrics.get("placement_rate"),
        "test_release_rate": test_metrics.get("release_rate"),
        "output_dir": training_summary.get("output_dir"),
        "best_model_path": training_summary.get("best_model_path"),
        "best_success_model_path": training_summary.get("best_success_model_path"),
        "training_summary_path": str(training_summary_path),
    }


def _write_results(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    target_success_rate: float,
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_success_rate": target_success_rate,
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


def _run_stage(
    *,
    base_config: ExperimentConfig,
    recipe: SweepRecipe,
    stage: str,
    init_checkpoint: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> TrainingArtifacts:
    config = _apply_recipe(
        base_config,
        recipe,
        stage=stage,
        output_root=output_dir / "runs" / recipe.name / stage,
    )
    if args.num_envs is not None:
        config.train.num_envs = args.num_envs

    if args.total_timesteps is not None:
        config.train.total_timesteps = args.total_timesteps
    elif stage == "warm" and args.warm_timesteps is not None:
        config.train.total_timesteps = args.warm_timesteps
    elif stage == "continue" and args.continue_timesteps is not None:
        config.train.total_timesteps = args.continue_timesteps

    return run_training(
        config,
        mode="contact",
        seed=config.train.seed,
        num_envs=config.train.num_envs,
        total_timesteps=config.train.total_timesteps,
        output_root=config.logging.output_root,
        wandb_mode=args.wandb_mode,
        init_checkpoint=init_checkpoint,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the warm-start arm clean-release sweep."
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Arm clean-release config matching the checkpoint action space, "
            "for example configs/arm_clean_release_joint.yaml."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        required=True,
        help="Local SAC checkpoint path for the iconic-haze-72 warm start.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Sweep ledger directory. Defaults to outputs/arm_clean_release_sweep/<timestamp>.",
    )
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--max-recipes", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--warm-timesteps", type=int, default=None)
    parser.add_argument("--continue-timesteps", type=int, default=500_000)
    parser.add_argument("--target-success-rate", type=float, default=0.10)
    parser.add_argument(
        "--stop-on-target",
        action="store_true",
        help=(
            "Stop after the first stage that reaches --target-success-rate. "
            "By default every recipe runs both warm and continue stages."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit zero even if strict validation success is not reached.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    init_checkpoint = _resolve_required_checkpoint(args.init_checkpoint)
    base_config = load_experiment_config(args.config)
    if base_config.env.embodiment != "arm_pinch":
        raise ValueError("The arm clean-release sweep requires env.embodiment=arm_pinch.")
    if base_config.env.task != "pick_place_ab":
        raise ValueError("The arm clean-release sweep requires env.task=pick_place_ab.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "arm_clean_release_sweep" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    recipes = DEFAULT_RECIPES[: args.max_recipes] if args.max_recipes else DEFAULT_RECIPES
    rows: list[dict[str, Any]] = []
    target_met = False

    for recipe in recipes:
        warm_artifacts = _run_stage(
            base_config=base_config,
            recipe=recipe,
            stage="warm",
            init_checkpoint=init_checkpoint,
            output_dir=output_dir,
            args=args,
        )
        warm_summary = _load_json(warm_artifacts.training_summary_path)
        warm_target_met = _meets_success_target(
            warm_summary.get("best_validation_metrics"),
            target_success_rate=args.target_success_rate,
        )
        rows.append(
            _result_row(
                recipe=recipe,
                stage="warm",
                init_checkpoint=init_checkpoint,
                training_summary=warm_summary,
                training_summary_path=warm_artifacts.training_summary_path,
                target_met=warm_target_met,
            )
        )
        _write_results(
            rows,
            output_dir,
            target_success_rate=args.target_success_rate,
        )
        if warm_target_met:
            target_met = True
        if _should_stop_after_stage(
            target_met=warm_target_met,
            stop_on_target=args.stop_on_target,
        ):
            break

        if not warm_artifacts.best_model_path.exists():
            raise FileNotFoundError(
                "Warm stage did not produce best_model.zip; refusing to continue "
                f"recipe {recipe.name}."
            )

        continue_checkpoint = str(warm_artifacts.best_model_path.resolve())
        continue_artifacts = _run_stage(
            base_config=base_config,
            recipe=recipe,
            stage="continue",
            init_checkpoint=continue_checkpoint,
            output_dir=output_dir,
            args=args,
        )
        continue_summary = _load_json(continue_artifacts.training_summary_path)
        continue_target_met = _meets_success_target(
            continue_summary.get("best_validation_metrics"),
            target_success_rate=args.target_success_rate,
        )
        rows.append(
            _result_row(
                recipe=recipe,
                stage="continue",
                init_checkpoint=continue_checkpoint,
                training_summary=continue_summary,
                training_summary_path=continue_artifacts.training_summary_path,
                target_met=continue_target_met,
            )
        )
        _write_results(
            rows,
            output_dir,
            target_success_rate=args.target_success_rate,
        )
        if continue_target_met:
            target_met = True
        if _should_stop_after_stage(
            target_met=continue_target_met,
            stop_on_target=args.stop_on_target,
        ):
            break

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "target_met": target_met,
                "target_success_rate": args.target_success_rate,
                "stop_on_target": args.stop_on_target,
                "results": rows,
            },
            indent=2,
        )
    )
    return 0 if target_met or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
