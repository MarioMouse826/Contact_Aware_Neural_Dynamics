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
from .env import ContactAwareGraspLiftEnv
from .evaluation import (
    compute_steps_to_80pct_success,
    evaluate_policy,
    save_json,
)
from .logging_utils import prepare_output_dir, resolve_run_id, start_wandb_run
from .modes import apply_mode_overrides, infer_mode_from_env_config, resolve_mode

MAX_PARALLEL_WORKERS = 12


@dataclass
class TrainingArtifacts:
    run_id: str
    mode: str
    output_dir: Path
    config_path: Path
    metadata_path: Path
    best_model_path: Path
    final_model_path: Path
    training_summary_path: Path


def _build_monitored_env(
    env_config_dict: dict[str, Any],
    reward_config_dict: dict[str, Any],
    seed: int,
) -> Monitor:
    wrapped_env = ContactAwareGraspLiftEnv(
        env_config=EnvConfig(**env_config_dict),
        reward_config=RewardConfig(**reward_config_dict),
    )
    wrapped_env.reset(seed=seed)
    return Monitor(
        wrapped_env,
        info_keywords=("is_success", "contact_stability", "max_lift_height"),
    )


def _build_eval_env(config: ExperimentConfig) -> ContactAwareGraspLiftEnv:
    return ContactAwareGraspLiftEnv(config.env, config.reward)


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
    eval_history: list[dict[str, Any]],
) -> dict[str, Any]:
    final_record = eval_history[-1] if eval_history else {}
    return {
        "mode": mode,
        "run_id": run_id,
        "seed": config.train.seed,
        "num_envs": config.train.num_envs,
        "output_dir": str(output_dir),
        "final_success_rate": final_record.get("success_rate"),
        "final_mean_reward": final_record.get("mean_reward"),
        "final_mean_contact_stability": final_record.get("mean_contact_stability"),
        "steps_to_80pct_success": compute_steps_to_80pct_success(eval_history),
        "evaluation_history": eval_history,
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

    run = start_wandb_run(
        config=run_config.to_dict(),
        logging_config=run_config.logging,
        job_type="train",
        tags=[mode, f"seed:{run_config.train.seed}", f"envs:{run_config.train.num_envs}"],
    )

    training_env = None
    eval_env = None
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
            "run_id": run_id,
            "seed": run_config.train.seed,
            "num_envs": run_config.train.num_envs,
            "wandb_entity": run_config.logging.wandb_entity,
            "wandb_project": run_config.logging.wandb_project,
        }
        save_json(metadata, metadata_path)

        training_env = _make_vector_env(run_config)
        eval_env = _build_eval_env(run_config)

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

        logger = configure_logger(str(logs_dir), ["stdout", "csv", "json", "tensorboard"])
        model.set_logger(logger)

        eval_callback = PeriodicEvalCallback(
            eval_env=eval_env,
            output_dir=output_dir,
            eval_freq=run_config.train.eval_freq,
            checkpoint_freq=run_config.train.checkpoint_freq,
            n_eval_episodes=run_config.eval.episodes,
            deterministic=run_config.eval.deterministic,
            eval_seed=run_config.train.seed + 10_000,
        )
        wandb_callback = WandbCallback(
            gradient_save_freq=run_config.logging.gradient_save_freq,
            model_save_freq=run_config.logging.model_save_freq,
            model_save_path=str(output_dir / "wandb_models"),
            verbose=0,
        )

        model.learn(
            total_timesteps=run_config.train.total_timesteps,
            callback=CallbackList([eval_callback, wandb_callback]),
            log_interval=None,
            progress_bar=False,
        )

        summary = _build_training_summary(
            mode=mode,
            run_id=run_id,
            config=run_config,
            output_dir=output_dir,
            eval_history=eval_callback.history,
        )
        save_json(summary, summary_path)

        return TrainingArtifacts(
            run_id=run_id,
            mode=mode,
            output_dir=output_dir,
            config_path=config_path,
            metadata_path=metadata_path,
            best_model_path=eval_callback.best_model_path,
            final_model_path=eval_callback.final_model_path,
            training_summary_path=summary_path,
        )
    finally:
        if training_env is not None:
            training_env.close()
        if eval_env is not None:
            eval_env.close()
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
    episodes: int | None = None,
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
    if episodes is not None:
        eval_config.eval.episodes = episodes

    eval_env = _build_eval_env(eval_config)
    try:
        model = SAC.load(str(checkpoint_path), env=eval_env, device=eval_config.train.device)
        summary = evaluate_policy(
            model,
            eval_env,
            n_episodes=eval_config.eval.episodes,
            deterministic=eval_config.eval.deterministic,
            base_seed=eval_config.train.seed + 20_000,
        )
    finally:
        eval_env.close()

    payload = {
        "requested_mode": mode,
        "training_mode": training_mode,
        "checkpoint_path": str(checkpoint_path),
        "episodes": eval_config.eval.episodes,
        **summary.to_dict(),
    }
    target_path = (
        Path(output_path)
        if output_path is not None
        else checkpoint_path.parent / f"evaluation_{mode}.json"
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
            float(result["final_success_rate"])
            for result in mode_results
            if result.get("final_success_rate") is not None
        ]
        steps_to_80 = [
            int(result["steps_to_80pct_success"])
            for result in mode_results
            if result.get("steps_to_80pct_success") is not None
        ]
        contact_stability = [
            float(result["final_mean_contact_stability"])
            for result in mode_results
            if result.get("final_mean_contact_stability") is not None
        ]
        aggregate[mode] = {
            "num_runs": len(mode_results),
            "mean_final_success_rate": (
                sum(success_rates) / len(success_rates) if success_rates else None
            ),
            "mean_steps_to_80pct_success": (
                sum(steps_to_80) / len(steps_to_80) if steps_to_80 else None
            ),
            "mean_final_contact_stability": (
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
        for mode in ("baseline", "contact", "always_contact"):
            artifacts = run_training(
                config,
                mode=mode,
                seed=seed,
                num_envs=num_envs,
                wandb_mode=wandb_mode,
            )
            training_summary = _load_json_if_exists(artifacts.training_summary_path)
            results.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "run_id": artifacts.run_id,
                    "final_success_rate": training_summary.get("final_success_rate"),
                    "steps_to_80pct_success": training_summary.get(
                        "steps_to_80pct_success"
                    ),
                    "final_mean_contact_stability": training_summary.get(
                        "final_mean_contact_stability"
                    ),
                }
            )

            if mode == "contact":
                ablation_payload = evaluate_checkpoint(
                    artifacts.best_model_path,
                    mode="contact_ablation",
                    output_path=suite_dir / f"contact_ablation_seed{seed}.json",
                )
                results.append(
                    {
                        "mode": "contact_ablation",
                        "seed": seed,
                        "run_id": artifacts.run_id,
                        "final_success_rate": ablation_payload.get("success_rate"),
                        "steps_to_80pct_success": None,
                        "final_mean_contact_stability": ablation_payload.get(
                            "mean_contact_stability"
                        ),
                    }
                )

    aggregate = aggregate_results(results)
    payload = {"results": results, "aggregate": aggregate}
    save_json(payload, suite_dir / "suite_results.json")
    _write_suite_csv(results, suite_dir / "suite_results.csv")
    return payload
