from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from .evaluation import compute_steps_to_80pct_success, evaluate_policy, save_json


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_env: Any,
        output_dir: str | Path,
        eval_freq: int,
        checkpoint_freq: int,
        n_eval_episodes: int,
        deterministic: bool,
        eval_seed: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.output_dir = Path(output_dir)
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.eval_seed = eval_seed

        self.history: list[dict[str, Any]] = []
        self.best_success_rate = float("-inf")
        self.best_mean_reward = float("-inf")
        self.last_eval_timestep = -1
        self.last_checkpoint_timestep = 0

        self.best_model_path = self.output_dir / "best_model.zip"
        self.final_model_path = self.output_dir / "final_model.zip"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save_history(self) -> None:
        save_json({"history": self.history}, self.output_dir / "evaluation_history.json")

    def _is_new_best(self, success_rate: float, mean_reward: float) -> bool:
        return (success_rate > self.best_success_rate) or (
            success_rate == self.best_success_rate and mean_reward > self.best_mean_reward
        )

    def _run_evaluation(self) -> None:
        summary = evaluate_policy(
            self.model,
            self.eval_env,
            n_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            base_seed=self.eval_seed,
            num_timesteps=self.num_timesteps,
        )
        record = summary.to_dict()
        self.history.append(record)
        self.last_eval_timestep = self.num_timesteps
        self._save_history()

        self.logger.record("eval/mean_reward", summary.mean_reward)
        self.logger.record("eval/success_rate", summary.success_rate)
        self.logger.record("eval/mean_episode_length", summary.mean_episode_length)
        self.logger.record("eval/mean_contact_stability", summary.mean_contact_stability)
        self.logger.record("eval/mean_max_lift_height", summary.mean_max_lift_height)
        self.logger.dump(self.num_timesteps)

        if self._is_new_best(summary.success_rate, summary.mean_reward):
            self.best_success_rate = summary.success_rate
            self.best_mean_reward = summary.mean_reward
            self.model.save(str(self.best_model_path))

    def _on_training_start(self) -> None:
        self._run_evaluation()

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self._run_evaluation()

        if (
            self.checkpoint_freq > 0
            and self.num_timesteps - self.last_checkpoint_timestep >= self.checkpoint_freq
        ):
            checkpoint_path = self.checkpoint_dir / f"model_{self.num_timesteps}.zip"
            self.model.save(str(checkpoint_path))
            self.last_checkpoint_timestep = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        if self.last_eval_timestep != self.num_timesteps:
            self._run_evaluation()
        self.model.save(str(self.final_model_path))

        summary_payload = {
            "final_success_rate": self.history[-1]["success_rate"] if self.history else None,
            "best_success_rate": self.best_success_rate if self.history else None,
            "steps_to_80pct_success": compute_steps_to_80pct_success(self.history),
        }
        save_json(summary_payload, self.output_dir / "eval_summary.json")
