from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .runtime import configure_runtime_environment

configure_runtime_environment()

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from .callbacks import PeriodicEvalCallback
from .config import EnvConfig, ExperimentConfig, RewardConfig, save_experiment_config
from .env import BaseContactAwareEnv, make_env
from .evaluation import (
    compute_steps_to_success_threshold,
    evaluate_policy,
    resolve_eval_split,
    save_json,
)
from .logging_utils import (
    prepare_output_dir,
    resolve_run_id,
    save_wandb_files,
    start_wandb_run,
)
from .modes import (
    apply_mode_overrides,
    infer_mode_from_env_config,
    resolve_mode,
    trainable_modes_for_env,
)

MAX_PARALLEL_WORKERS = 12


def _config_type_tag(env_config: EnvConfig) -> str:
    config_type_by_embodiment = {
        "cartesian_gripper": "cartesian",
        "arm_pinch": "arm",
    }
    return config_type_by_embodiment.get(env_config.embodiment, env_config.embodiment)


def _build_wandb_tags(config: ExperimentConfig, *, mode: str) -> list[str]:
    tags = [
        mode,
        f"task:{config.env.task}",
        f"config:{_config_type_tag(config.env)}",
        f"seed:{config.train.seed}",
        f"envs:{config.train.num_envs}",
    ]
    tags.extend(str(tag) for tag in config.logging.wandb_tags)
    return tags


@dataclass
class TrainingArtifacts:
    run_id: str
    mode: str
    output_dir: Path
    config_path: Path
    metadata_path: Path
    best_model_path: Path
    best_success_model_path: Path
    latest_model_path: Path
    final_model_path: Path
    training_summary_path: Path


def _build_monitored_env(
    env_config_dict: dict[str, Any],
    reward_config_dict: dict[str, Any],
    seed: int,
) -> Monitor:
    wrapped_env = make_env(
        env_config=EnvConfig(**env_config_dict),
        reward_config=RewardConfig(**reward_config_dict),
    )
    wrapped_env.reset(seed=seed)
    return Monitor(
        wrapped_env,
        info_keywords=(
            "is_success",
            "contact_stability",
            "dual_contact_stability",
            "max_lift_height",
            "best_success_streak",
            "threshold_crossed",
            "steps_above_success_height",
            "goal_distance_xy",
            "best_goal_distance_xy",
            "episode_has_grasped",
            "episode_has_lifted_grasp",
            "episode_has_lifted_for_transport",
            "episode_has_over_goal",
            "episode_has_placed",
            "episode_has_released",
            "episode_has_settled",
            "is_placed",
            "is_released",
            "is_settled",
        ),
    )


def _build_eval_env(config: ExperimentConfig) -> BaseContactAwareEnv:
    return make_env(config.env, config.reward)


def _make_vector_env(config: ExperimentConfig) -> DummyVecEnv | SubprocVecEnv:
    num_envs = config.train.num_envs
    if num_envs < 1 or num_envs > MAX_PARALLEL_WORKERS:
        raise ValueError(
            f"num_envs must be between 1 and {MAX_PARALLEL_WORKERS}, got {num_envs}"
        )

    env_payload = config.to_dict()
    env_fns = [
        partial(
            _build_monitored_env,
            env_payload["env"],
            env_payload["reward"],
            config.train.seed + env_index,
        )
        for env_index in range(num_envs)
    ]

    if num_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="spawn")


def _build_training_summary(
    *,
    mode: str,
    run_id: str,
    config: ExperimentConfig,
    output_dir: Path,
    eval_callback: PeriodicEvalCallback,
    test_summary: dict[str, Any] | None,
    init_checkpoint: str | None,
) -> dict[str, Any]:
    best_model_path = (
        str(eval_callback.best_model_path) if eval_callback.best_model_path.exists() else None
    )
    best_success_model_path = (
        str(eval_callback.best_success_model_path)
        if eval_callback.best_success_model_path.exists()
        else None
    )
    return {
        "mode": mode,
        "embodiment": config.env.embodiment,
        "run_id": run_id,
        "seed": config.train.seed,
        "num_envs": config.train.num_envs,
        "output_dir": str(output_dir),
        "init_checkpoint_path": init_checkpoint,
        "training_status": eval_callback.training_status,
        "stop_reason": eval_callback.stop_reason,
        "best_model_path": best_model_path,
        "best_timestep": eval_callback.best_timestep,
        "best_success_model_path": best_success_model_path,
        "latest_model_path": str(eval_callback.latest_model_path),
        "final_model_path": str(eval_callback.final_model_path),
        "best_success_timestep": eval_callback.best_success_timestep,
        "best_validation_metrics": eval_callback.best_validation_summary,
        "best_success_validation_metrics": eval_callback.best_success_validation_summary,
        "final_monitor_metrics": eval_callback.final_monitor_summary,
        "final_validation_metrics": eval_callback.final_validation_summary,
        "monitor_history_path": str(output_dir / "monitor_history.json"),
        "validation_history_path": str(output_dir / "validation_history.json"),
        "test_metrics": test_summary,
        "steps_to_target_validation_success": compute_steps_to_success_threshold(
            eval_callback.validation_history,
            threshold=config.train.early_stop_success_rate,
        ),
    }


def _write_suite_csv(results: list[dict[str, Any]], path: Path) -> None:
    if not results:
        return
    fieldnames = sorted({key for result in results for key in result})
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def run_training(
    config: ExperimentConfig,
    *,
    mode: str,
    seed: int | None = None,
    num_envs: int | None = None,
    total_timesteps: int | None = None,
    output_root: str | None = None,
    wandb_mode: str | None = None,
    init_checkpoint: str | Path | None = None,
) -> TrainingArtifacts:
    resolved_mode = resolve_mode(mode)
    if not resolved_mode.trainable:
        raise ValueError(f"Mode '{mode}' is evaluation-only and cannot be trained.")

    run_config = apply_mode_overrides(config, mode)
    if seed is not None:
        run_config.train.seed = seed
    if num_envs is not None:
        run_config.train.num_envs = num_envs
    if total_timesteps is not None:
        run_config.train.total_timesteps = total_timesteps
    if output_root is not None:
        run_config.logging.output_root = output_root
    if wandb_mode is not None:
        run_config.logging.wandb_mode = wandb_mode
    resolved_init_checkpoint = None
    if init_checkpoint is not None:
        init_checkpoint_path = Path(init_checkpoint).expanduser().resolve()
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Initial checkpoint does not exist: {init_checkpoint_path}"
            )
        resolved_init_checkpoint = str(init_checkpoint_path)

    wandb_config = run_config.to_dict()
    wandb_config["init_checkpoint_path"] = resolved_init_checkpoint

    run = start_wandb_run(
        config=wandb_config,
        logging_config=run_config.logging,
        job_type="train",
        tags=_build_wandb_tags(run_config, mode=mode),
    )

    training_env = None
    monitor_env = None
    validation_env = None
    try:
        run_id = resolve_run_id(run)
        output_dir = prepare_output_dir(run_config.logging.output_root, run_id)
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / "config.yaml"
        metadata_path = output_dir / "metadata.json"
        summary_path = output_dir / "training_summary.json"
        save_experiment_config(run_config, config_path)

        metadata = {
            "mode": mode,
            "embodiment": run_config.env.embodiment,
            "run_id": run_id,
            "seed": run_config.train.seed,
            "num_envs": run_config.train.num_envs,
            "init_checkpoint_path": resolved_init_checkpoint,
            "wandb_entity": run_config.logging.wandb_entity,
            "wandb_project": run_config.logging.wandb_project,
        }
        save_json(metadata, metadata_path)

        training_env = _make_vector_env(run_config)
        monitor_env = _build_eval_env(run_config)
        validation_env = _build_eval_env(run_config)

        if resolved_init_checkpoint is None:
            model = SAC(
                "MlpPolicy",
                training_env,
                learning_rate=run_config.train.learning_rate,
                buffer_size=run_config.train.buffer_size,
                learning_starts=run_config.train.learning_starts,
                batch_size=run_config.train.batch_size,
                gamma=run_config.train.gamma,
                tau=run_config.train.tau,
                train_freq=(run_config.train.train_freq, "step"),
                gradient_steps=run_config.train.gradient_steps,
                policy_kwargs={"net_arch": list(run_config.train.net_arch)},
                seed=run_config.train.seed,
                verbose=1,
                device=run_config.train.device,
            )
        else:
            model = SAC.load(
                resolved_init_checkpoint,
                env=training_env,
                device=run_config.train.device,
            )
            model.set_random_seed(run_config.train.seed)

        logger = configure_logger(str(logs_dir), ["stdout", "csv", "json", "tensorboard"])
        model.set_logger(logger)

        monitor_split, monitor_episodes, monitor_seed = resolve_eval_split(
            run_config,
            split="monitor",
        )
        validation_split, validation_episodes, validation_seed = resolve_eval_split(
            run_config,
            split="validation",
        )
        if monitor_split != "monitor" or validation_split != "validation":
            raise ValueError("Failed to resolve named evaluation splits.")

        eval_callback = PeriodicEvalCallback(
            monitor_env=monitor_env,
            validation_env=validation_env,
            output_dir=output_dir,
            total_timesteps=run_config.train.total_timesteps,
            eval_freq=run_config.train.eval_freq,
            checkpoint_freq=run_config.train.checkpoint_freq,
            monitor_episodes=monitor_episodes,
            monitor_seed=monitor_seed,
            validation_episodes=validation_episodes,
            validation_seed=validation_seed,
            deterministic=run_config.eval.deterministic,
            early_stop_success_rate=run_config.train.early_stop_success_rate,
            early_stop_success_patience=run_config.train.early_stop_success_patience,
            early_stop_plateau_patience=run_config.train.early_stop_plateau_patience,
            early_stop_plateau_start_timesteps=(
                run_config.train.early_stop_plateau_start_timesteps
            ),
        )
        wandb_callback = WandbCallback(
            gradient_save_freq=run_config.logging.gradient_save_freq,
            model_save_freq=0,
            verbose=0,
        )

        model.learn(
            total_timesteps=run_config.train.total_timesteps,
            callback=CallbackList([eval_callback, wandb_callback]),
            log_interval=None,
            progress_bar=False,
        )

        test_summary = None
        if eval_callback.best_model_path.exists():
            test_summary = evaluate_checkpoint(
                eval_callback.best_model_path,
                mode=mode,
                config_path=config_path,
                split="test",
                output_path=output_dir / "test_summary.json",
            )

        summary = _build_training_summary(
            mode=mode,
            run_id=run_id,
            config=run_config,
            output_dir=output_dir,
            eval_callback=eval_callback,
            test_summary=test_summary,
            init_checkpoint=resolved_init_checkpoint,
        )
        save_json(summary, summary_path)
        save_wandb_files(
            [
                eval_callback.best_model_path,
                eval_callback.latest_model_path,
            ],
            base_path=output_dir,
        )

        return TrainingArtifacts(
            run_id=run_id,
            mode=mode,
            output_dir=output_dir,
            config_path=config_path,
            metadata_path=metadata_path,
            best_model_path=eval_callback.best_model_path,
            best_success_model_path=eval_callback.best_success_model_path,
            latest_model_path=eval_callback.latest_model_path,
            final_model_path=eval_callback.final_model_path,
            training_summary_path=summary_path,
        )
    finally:
        if training_env is not None:
            training_env.close()
        if monitor_env is not None:
            monitor_env.close()
        if validation_env is not None:
            validation_env.close()
        if hasattr(run, "finish"):
            run.finish()


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    mode: str,
    config_path: str | Path | None = None,
    split: str = "validation",
    episodes: int | None = None,
    base_seed: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if config_path is None:
        config_path = checkpoint_path.parent / "config.yaml"

    from .config import load_experiment_config

    config = load_experiment_config(config_path)
    metadata = _load_json_if_exists(checkpoint_path.parent / "metadata.json")
    training_mode = metadata.get("mode", infer_mode_from_env_config(config))
    resolve_mode(mode)

    valid_modes = {training_mode}
    if training_mode == "contact":
        valid_modes.add("contact_ablation")
    if mode not in valid_modes:
        raise ValueError(
            f"Checkpoint trained in mode '{training_mode}' cannot be evaluated with mode '{mode}'."
        )

    eval_config = apply_mode_overrides(config, mode)
    resolved_split, resolved_episodes, resolved_base_seed = resolve_eval_split(
        eval_config,
        split=split,
        episodes=episodes,
        base_seed=base_seed,
    )

    eval_env = _build_eval_env(eval_config)
    try:
        model = SAC.load(str(checkpoint_path), env=eval_env, device=eval_config.train.device)
        summary = evaluate_policy(
            model,
            eval_env,
            n_episodes=resolved_episodes,
            deterministic=eval_config.eval.deterministic,
            base_seed=resolved_base_seed,
            split=resolved_split,
        )
    finally:
        eval_env.close()

    payload = {
        "requested_mode": mode,
        "training_mode": training_mode,
        "checkpoint_path": str(checkpoint_path),
        "split": resolved_split,
        "base_seed": resolved_base_seed,
        "episodes": resolved_episodes,
        **summary.to_dict(),
    }
    target_path = (
        Path(output_path)
        if output_path is not None
        else checkpoint_path.parent / f"evaluation_{mode}_{resolved_split}.json"
    )
    save_json(payload, target_path)
    return payload


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["mode"], []).append(result)

    aggregate: dict[str, Any] = {}
    for mode, mode_results in grouped.items():
        success_rates = [
            float(result["success_rate"])
            for result in mode_results
            if result.get("success_rate") is not None
        ]
        contact_stability = [
            float(result["mean_contact_stability"])
            for result in mode_results
            if result.get("mean_contact_stability") is not None
        ]
        aggregate[mode] = {
            "num_runs": len(mode_results),
            "mean_success_rate": (
                sum(success_rates) / len(success_rates) if success_rates else None
            ),
            "mean_contact_stability": (
                sum(contact_stability) / len(contact_stability)
                if contact_stability
                else None
            ),
        }
    return aggregate


def run_proposal_suite(
    config: ExperimentConfig,
    *,
    seeds: list[int],
    num_envs: int,
    output_dir: str | Path,
    wandb_mode: str | None = None,
) -> dict[str, Any]:
    suite_dir = Path(output_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for seed in seeds:
        for mode in trainable_modes_for_env(config.env):
            artifacts = run_training(
                config,
                mode=mode,
                seed=seed,
                num_envs=num_envs,
                wandb_mode=wandb_mode,
            )
            training_summary = _load_json_if_exists(artifacts.training_summary_path)
            best_validation_metrics = training_summary.get("best_validation_metrics") or {}
            results.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "run_id": artifacts.run_id,
                    "training_status": training_summary.get("training_status"),
                    "success_rate": best_validation_metrics.get("success_rate"),
                    "mean_contact_stability": best_validation_metrics.get(
                        "mean_contact_stability"
                    ),
                }
            )

            if mode == "contact" and artifacts.best_success_model_path.exists():
                ablation_payload = evaluate_checkpoint(
                    artifacts.best_success_model_path,
                    mode="contact_ablation",
                    split="validation",
                    output_path=suite_dir / f"contact_ablation_seed{seed}.json",
                )
                results.append(
                    {
                        "mode": "contact_ablation",
                        "seed": seed,
                        "run_id": artifacts.run_id,
                        "training_status": training_summary.get("training_status"),
                        "success_rate": ablation_payload.get("success_rate"),
                        "mean_contact_stability": ablation_payload.get(
                            "mean_contact_stability"
                        ),
                    }
                )
            elif mode == "contact":
                results.append(
                    {
                        "mode": "contact_ablation",
                        "seed": seed,
                        "run_id": artifacts.run_id,
                        "training_status": "skipped_no_success_checkpoint",
                        "success_rate": None,
                        "mean_contact_stability": None,
                    }
                )

    aggregate = aggregate_results(results)
    payload = {"results": results, "aggregate": aggregate}
    save_json(payload, suite_dir / "suite_results.json")
    _write_suite_csv(results, suite_dir / "suite_results.csv")
    return payload
