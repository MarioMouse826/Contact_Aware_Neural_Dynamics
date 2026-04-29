from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from .evaluation import EvaluationSummary, evaluate_policy, save_json


class PeriodicEvalCallback(BaseCallback):
    """Evaluate monitor/validation splits and manage best/latest checkpoints."""

    def __init__(
        self,
        *,
        monitor_env: Any,
        validation_env: Any,
        output_dir: str | Path,
        total_timesteps: int,
        eval_freq: int,
        checkpoint_freq: int,
        monitor_episodes: int,
        monitor_seed: int,
        validation_episodes: int,
        validation_seed: int,
        deterministic: bool,
        early_stop_success_rate: float,
        early_stop_success_patience: int,
        early_stop_plateau_patience: int,
        early_stop_plateau_start_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.monitor_env = monitor_env
        self.validation_env = validation_env
        self.output_dir = Path(output_dir)
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.monitor_episodes = monitor_episodes
        self.monitor_seed = monitor_seed
        self.validation_episodes = validation_episodes
        self.validation_seed = validation_seed
        self.deterministic = deterministic
        self.early_stop_success_rate = early_stop_success_rate
        self.early_stop_success_patience = early_stop_success_patience
        self.early_stop_plateau_patience = early_stop_plateau_patience
        self.early_stop_plateau_start_timesteps = early_stop_plateau_start_timesteps

        self.monitor_history: list[dict[str, Any]] = []
        self.validation_history: list[dict[str, Any]] = []
        self.best_monitor_tuple: tuple[float, ...] | None = None
        self.best_validation_tuple: tuple[float, ...] | None = None
        self.best_success_validation_tuple: tuple[float, ...] | None = None
        self.best_validation_summary: dict[str, Any] | None = None
        self.best_timestep: int | None = None
        self.best_success_validation_summary: dict[str, Any] | None = None
        self.best_success_timestep: int | None = None
        self.final_monitor_summary: dict[str, Any] | None = None
        self.final_validation_summary: dict[str, Any] | None = None
        self.training_status = "running"
        self.stop_reason: str | None = None

        self.last_monitor_timestep = -1
        self.last_validation_timestep = -1
        self.last_checkpoint_timestep = 0
        self.consecutive_target_hits = 0
        self.validation_plateau_count = 0
        self.should_stop_training = False

        self.best_model_path = self.output_dir / "best_model.zip"
        self.best_success_model_path = self.output_dir / "best_success_model.zip"
        self.latest_model_path = self.output_dir / "latest_model.zip"
        self.final_model_path = self.output_dir / "final_model.zip"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save_history(self, split: str, history: list[dict[str, Any]]) -> None:
        save_json({"split": split, "history": history}, self.output_dir / f"{split}_history.json")

    def _priority(self, summary: EvaluationSummary) -> tuple[float, ...]:
        timestep = int(summary.num_timesteps or 0)
        if summary.task == "pick_place_ab":
            return (
                float(summary.success_rate),
                float(summary.settle_rate),
                float(summary.release_rate),
                float(summary.placement_rate),
                float(summary.over_goal_rate),
                float(summary.transport_ready_rate),
                float(summary.lifted_grasp_rate),
                -float(summary.mean_best_goal_distance_xy),
                -float(summary.mean_episode_length),
                -float(timestep),
            )
        return (
            float(summary.success_rate),
            float(summary.mean_best_success_streak),
            -float(summary.mean_episode_length),
            -float(timestep),
        )

    def _is_better(
        self,
        summary: EvaluationSummary,
        best_priority: tuple[float, ...] | None,
    ) -> bool:
        if best_priority is None:
            return True
        return self._priority(summary) > best_priority

    def _record_summary(self, prefix: str, summary: EvaluationSummary) -> None:
        self.logger.record(f"{prefix}/success_rate", summary.success_rate)
        self.logger.record(f"{prefix}/near_success_rate", summary.near_success_rate)
        self.logger.record(f"{prefix}/threshold_cross_rate", summary.threshold_cross_rate)
        self.logger.record(
            f"{prefix}/mean_best_success_streak", summary.mean_best_success_streak
        )
        self.logger.record(f"{prefix}/mean_reward", summary.mean_reward)
        self.logger.record(f"{prefix}/mean_episode_length", summary.mean_episode_length)
        self.logger.record(
            f"{prefix}/mean_contact_stability", summary.mean_contact_stability
        )
        self.logger.record(
            f"{prefix}/mean_dual_contact_stability",
            summary.mean_dual_contact_stability,
        )
        self.logger.record(f"{prefix}/mean_max_lift_height", summary.mean_max_lift_height)
        self.logger.record(f"{prefix}/mean_goal_distance_xy", summary.mean_goal_distance_xy)
        self.logger.record(
            f"{prefix}/mean_best_goal_distance_xy", summary.mean_best_goal_distance_xy
        )
        self.logger.record(f"{prefix}/grasp_rate", summary.grasp_rate)
        self.logger.record(f"{prefix}/lifted_grasp_rate", summary.lifted_grasp_rate)
        self.logger.record(f"{prefix}/transport_ready_rate", summary.transport_ready_rate)
        self.logger.record(f"{prefix}/over_goal_rate", summary.over_goal_rate)
        self.logger.record(f"{prefix}/placement_rate", summary.placement_rate)
        self.logger.record(f"{prefix}/release_rate", summary.release_rate)
        self.logger.record(f"{prefix}/settle_rate", summary.settle_rate)
        self.logger.record(f"{prefix}/num_episodes", summary.num_episodes)
        for reason, count in summary.termination_reason_counts.items():
            self.logger.record(f"{prefix}/termination_{reason}", float(count))
        self.logger.dump(self.num_timesteps)

    def _evaluate_split(
        self,
        *,
        split: str,
        env: Any,
        n_episodes: int,
        base_seed: int,
        trigger: str,
    ) -> EvaluationSummary:
        summary = evaluate_policy(
            self.model,
            env,
            n_episodes=n_episodes,
            deterministic=self.deterministic,
            base_seed=base_seed,
            num_timesteps=self.num_timesteps,
            split=split,
        )
        record = summary.to_dict()
        record["trigger"] = trigger
        if split == "monitor":
            self.monitor_history.append(record)
            self.final_monitor_summary = record
            self.last_monitor_timestep = self.num_timesteps
            self._save_history("monitor", self.monitor_history)
        else:
            self.validation_history.append(record)
            self.final_validation_summary = record
            self.last_validation_timestep = self.num_timesteps
            self._save_history("validation", self.validation_history)
        self._record_summary(split, summary)
        return summary

    def _run_validation(self, *, trigger: str) -> EvaluationSummary:
        summary = self._evaluate_split(
            split="validation",
            env=self.validation_env,
            n_episodes=self.validation_episodes,
            base_seed=self.validation_seed,
            trigger=trigger,
        )
        improved = self._is_better(summary, self.best_validation_tuple)
        if improved:
            self.best_validation_tuple = self._priority(summary)
            self.best_validation_summary = summary.to_dict()
            self.best_timestep = int(summary.num_timesteps or 0)
            self.model.save(str(self.best_model_path))

        if summary.success_rate >= self.early_stop_success_rate:
            self.consecutive_target_hits += 1
        else:
            self.consecutive_target_hits = 0

        if improved:
            self.validation_plateau_count = 0
        elif self.num_timesteps >= self.early_stop_plateau_start_timesteps:
            self.validation_plateau_count += 1

        if summary.success_rate > 0.0 and self._is_better(
            summary, self.best_success_validation_tuple
        ):
            self.best_success_validation_tuple = self._priority(summary)
            self.best_success_validation_summary = summary.to_dict()
            self.best_success_timestep = int(summary.num_timesteps or 0)
            self.model.save(str(self.best_success_model_path))

        if (
            self.consecutive_target_hits >= self.early_stop_success_patience
            and self.stop_reason is None
        ):
            self.stop_reason = "target_success_reached"
            self.should_stop_training = True
        elif (
            self.validation_plateau_count >= self.early_stop_plateau_patience
            and self.stop_reason is None
        ):
            self.stop_reason = "validation_plateau"
            self.should_stop_training = True

        return summary

    def _on_training_start(self) -> None:
        monitor_summary = self._evaluate_split(
            split="monitor",
            env=self.monitor_env,
            n_episodes=self.monitor_episodes,
            base_seed=self.monitor_seed,
            trigger="training_start",
        )
        self.best_monitor_tuple = self._priority(monitor_summary)
        self._run_validation(trigger="training_start")

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps - self.last_monitor_timestep >= self.eval_freq:
            monitor_summary = self._evaluate_split(
                split="monitor",
                env=self.monitor_env,
                n_episodes=self.monitor_episodes,
                base_seed=self.monitor_seed,
                trigger="periodic",
            )
            if self._is_better(monitor_summary, self.best_monitor_tuple):
                self.best_monitor_tuple = self._priority(monitor_summary)
            self._run_validation(trigger="periodic")

        if (
            self.checkpoint_freq > 0
            and self.num_timesteps - self.last_checkpoint_timestep >= self.checkpoint_freq
        ):
            checkpoint_path = self.checkpoint_dir / f"model_{self.num_timesteps}.zip"
            self.model.save(str(checkpoint_path))
            self.last_checkpoint_timestep = self.num_timesteps

        return not self.should_stop_training

    def _on_training_end(self) -> None:
        if self.last_monitor_timestep != self.num_timesteps:
            self._evaluate_split(
                split="monitor",
                env=self.monitor_env,
                n_episodes=self.monitor_episodes,
                base_seed=self.monitor_seed,
                trigger="training_end",
            )
        if self.last_validation_timestep != self.num_timesteps:
            self._run_validation(trigger="training_end")

        self.model.save(str(self.final_model_path))
        self.model.save(str(self.latest_model_path))

        if self.stop_reason is None:
            if self.num_timesteps >= self.total_timesteps:
                self.stop_reason = "max_timesteps_reached"
            else:
                self.stop_reason = "training_ended"

        if self.best_success_validation_summary is None:
            self.training_status = "no_success_checkpoint"
        else:
            self.training_status = "success_checkpoint_selected"
