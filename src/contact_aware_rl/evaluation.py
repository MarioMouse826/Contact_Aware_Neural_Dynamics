from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import gymnasium as gym

from .config import ExperimentConfig

DEFAULT_EVAL_SPLIT = "validation"
NAMED_EVAL_SPLITS = ("monitor", "validation", "test")
EVAL_SPLITS = NAMED_EVAL_SPLITS + ("custom",)


@dataclass(frozen=True)
class EvaluationSummary:
    task: str
    split: str
    base_seed: int
    num_timesteps: int | None
    num_episodes: int
    mean_reward: float
    success_rate: float
    near_success_rate: float
    threshold_cross_rate: float
    mean_best_success_streak: float
    mean_episode_length: float
    mean_contact_stability: float
    mean_dual_contact_stability: float
    mean_max_lift_height: float
    mean_goal_distance_xy: float
    mean_best_goal_distance_xy: float
    grasp_rate: float
    lifted_grasp_rate: float
    transport_ready_rate: float
    over_goal_rate: float
    placement_rate: float
    release_rate: float
    settle_rate: float
    termination_reason_counts: dict[str, int]
    episodes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_eval_split(
    config: ExperimentConfig,
    *,
    split: str = DEFAULT_EVAL_SPLIT,
    episodes: int | None = None,
    base_seed: int | None = None,
) -> tuple[str, int, int]:
    if split not in EVAL_SPLITS:
        raise ValueError(f"Unsupported evaluation split: {split}")

    if split == "custom":
        if base_seed is None:
            raise ValueError("A custom evaluation split requires an explicit base_seed.")
        return split, int(episodes or config.eval.validation.episodes), int(base_seed)

    if base_seed is not None:
        raise ValueError("base_seed can only be overridden for the custom evaluation split.")

    split_config = getattr(config.eval, split)
    resolved_episodes = int(episodes or split_config.episodes)
    resolved_base_seed = int(config.train.seed + split_config.seed_offset)
    return split, resolved_episodes, resolved_base_seed


def summarize_episodes(
    episodes: list[dict[str, Any]],
    *,
    task: str,
    split: str,
    base_seed: int,
    num_timesteps: int | None = None,
    near_success_threshold: int = 1,
) -> EvaluationSummary:
    if not episodes:
        raise ValueError("At least one evaluation episode is required.")

    termination_counts = Counter(
        str(episode.get("termination_reason", "unknown")) for episode in episodes
    )

    return EvaluationSummary(
        task=task,
        split=split,
        base_seed=base_seed,
        num_timesteps=num_timesteps,
        num_episodes=len(episodes),
        mean_reward=mean(float(episode["reward"]) for episode in episodes),
        success_rate=mean(float(episode.get("is_success", 0.0)) for episode in episodes),
        near_success_rate=mean(
            float(int(int(episode.get("best_success_streak", 0)) >= near_success_threshold))
            for episode in episodes
        ),
        threshold_cross_rate=mean(
            float(episode.get("threshold_crossed", 0.0)) for episode in episodes
        ),
        mean_best_success_streak=mean(
            float(episode.get("best_success_streak", 0.0)) for episode in episodes
        ),
        mean_episode_length=mean(float(episode.get("length", 0.0)) for episode in episodes),
        mean_contact_stability=mean(
            float(episode.get("contact_stability", 0.0)) for episode in episodes
        ),
        mean_dual_contact_stability=mean(
            float(episode.get("dual_contact_stability", 0.0)) for episode in episodes
        ),
        mean_max_lift_height=mean(
            float(episode.get("max_lift_height", 0.0)) for episode in episodes
        ),
        mean_goal_distance_xy=mean(
            float(episode.get("goal_distance_xy", 0.0)) for episode in episodes
        ),
        mean_best_goal_distance_xy=mean(
            float(episode.get("best_goal_distance_xy", 0.0)) for episode in episodes
        ),
        grasp_rate=mean(float(episode.get("episode_has_grasped", 0.0)) for episode in episodes),
        lifted_grasp_rate=mean(
            float(episode.get("episode_has_lifted_grasp", 0.0)) for episode in episodes
        ),
        transport_ready_rate=mean(
            float(episode.get("episode_has_lifted_for_transport", 0.0))
            for episode in episodes
        ),
        over_goal_rate=mean(
            float(episode.get("episode_has_over_goal", 0.0)) for episode in episodes
        ),
        placement_rate=mean(float(episode.get("is_placed", 0.0)) for episode in episodes),
        release_rate=mean(float(episode.get("is_released", 0.0)) for episode in episodes),
        settle_rate=mean(float(episode.get("is_settled", 0.0)) for episode in episodes),
        termination_reason_counts=dict(sorted(termination_counts.items())),
        episodes=episodes,
    )


def evaluate_policy(
    model: Any,
    env: gym.Env,
    *,
    n_episodes: int,
    deterministic: bool = True,
    base_seed: int = 0,
    num_timesteps: int | None = None,
    split: str = DEFAULT_EVAL_SPLIT,
) -> EvaluationSummary:
    episodes: list[dict[str, Any]] = []
    task = str(getattr(getattr(env, "env_config", None), "task", "unknown"))
    near_success_threshold = 1
    if hasattr(env, "success_hold_steps"):
        near_success_threshold = max(1, math.ceil(int(env.success_hold_steps) / 2))
    elif hasattr(env, "env_config"):
        near_success_threshold = max(1, math.ceil(env.env_config.success_hold_steps / 2))

    for episode_index in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + episode_index)
        done = False
        total_reward = 0.0
        length = 0
        final_info: dict[str, Any] = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            length += 1
            final_info = info
            done = bool(terminated or truncated)

        episodes.append(
            {
                "episode_index": episode_index,
                "reward": total_reward,
                "length": length,
                "is_success": float(final_info.get("is_success", 0.0)),
                "best_success_streak": int(final_info.get("best_success_streak", 0)),
                "threshold_crossed": float(final_info.get("threshold_crossed", 0.0)),
                "steps_above_success_height": int(
                    final_info.get("steps_above_success_height", 0)
                ),
                "contact_stability": float(final_info.get("contact_stability", 0.0)),
                "dual_contact_stability": float(
                    final_info.get("dual_contact_stability", 0.0)
                ),
                "max_lift_height": float(final_info.get("max_lift_height", 0.0)),
                "goal_distance_xy": float(final_info.get("goal_distance_xy", 0.0)),
                "best_goal_distance_xy": float(final_info.get("best_goal_distance_xy", 0.0)),
                "episode_has_grasped": float(final_info.get("episode_has_grasped", 0.0)),
                "episode_has_lifted_grasp": float(
                    final_info.get("episode_has_lifted_grasp", 0.0)
                ),
                "episode_has_lifted_for_transport": float(
                    final_info.get("episode_has_lifted_for_transport", 0.0)
                ),
                "episode_has_over_goal": float(final_info.get("episode_has_over_goal", 0.0)),
                "is_placed": float(final_info.get("episode_has_placed", 0.0)),
                "is_released": float(final_info.get("episode_has_released", 0.0)),
                "is_settled": float(final_info.get("episode_has_settled", 0.0)),
                "termination_reason": final_info.get("termination_reason", "unknown"),
            }
        )

    return summarize_episodes(
        episodes,
        task=task,
        split=split,
        base_seed=base_seed,
        num_timesteps=num_timesteps,
        near_success_threshold=near_success_threshold,
    )


def compute_steps_to_success_threshold(
    history: list[dict[str, Any]], threshold: float = 0.8
) -> int | None:
    for record in history:
        if float(record.get("success_rate", 0.0)) >= threshold:
            return int(record["num_timesteps"])
    return None


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
