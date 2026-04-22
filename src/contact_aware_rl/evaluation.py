from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class EvaluationSummary:
    num_timesteps: int | None
    mean_reward: float
    success_rate: float
    mean_episode_length: float
    mean_contact_stability: float
    mean_max_lift_height: float
    episodes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_policy(
    model: Any,
    env: gym.Env,
    *,
    n_episodes: int,
    deterministic: bool = True,
    base_seed: int = 0,
    num_timesteps: int | None = None,
) -> EvaluationSummary:
    episodes: list[dict[str, Any]] = []

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
                "contact_stability": float(final_info.get("contact_stability", 0.0)),
                "max_lift_height": float(final_info.get("max_lift_height", 0.0)),
                "termination_reason": final_info.get("termination_reason"),
            }
        )

    return EvaluationSummary(
        num_timesteps=num_timesteps,
        mean_reward=mean(episode["reward"] for episode in episodes),
        success_rate=mean(episode["is_success"] for episode in episodes),
        mean_episode_length=mean(episode["length"] for episode in episodes),
        mean_contact_stability=mean(
            episode["contact_stability"] for episode in episodes
        ),
        mean_max_lift_height=mean(episode["max_lift_height"] for episode in episodes),
        episodes=episodes,
    )


def compute_steps_to_80pct_success(
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
