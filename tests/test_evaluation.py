from __future__ import annotations

import pytest

from contact_aware_rl.config import ExperimentConfig
from contact_aware_rl.evaluation import resolve_eval_split, summarize_episodes


def test_resolve_eval_split_uses_named_defaults_and_custom_override() -> None:
    config = ExperimentConfig()

    split, episodes, base_seed = resolve_eval_split(config, split="validation")
    assert split == "validation"
    assert episodes == 100
    assert base_seed == 20_000

    custom_split, custom_episodes, custom_base_seed = resolve_eval_split(
        config,
        split="custom",
        episodes=7,
        base_seed=321,
    )
    assert custom_split == "custom"
    assert custom_episodes == 7
    assert custom_base_seed == 321

    with pytest.raises(ValueError):
        resolve_eval_split(config, split="monitor", base_seed=123)


def test_summarize_episodes_reports_streak_and_termination_metrics() -> None:
    summary = summarize_episodes(
        [
            {
                "reward": 10.0,
                "length": 20,
                "is_success": 1.0,
                "best_success_streak": 10,
                "threshold_crossed": 1.0,
                "steps_above_success_height": 10,
                "contact_stability": 0.8,
                "dual_contact_stability": 0.7,
                "max_lift_height": 0.11,
                "goal_distance_xy": 0.01,
                "best_goal_distance_xy": 0.01,
                "episode_has_grasped": 1.0,
                "episode_has_lifted_grasp": 1.0,
                "is_placed": 1.0,
                "is_released": 1.0,
                "is_settled": 1.0,
                "termination_reason": "success",
            },
            {
                "reward": 4.0,
                "length": 50,
                "is_success": 0.0,
                "best_success_streak": 5,
                "threshold_crossed": 1.0,
                "steps_above_success_height": 8,
                "contact_stability": 0.6,
                "dual_contact_stability": 0.4,
                "max_lift_height": 0.09,
                "goal_distance_xy": 0.03,
                "best_goal_distance_xy": 0.02,
                "episode_has_grasped": 1.0,
                "episode_has_lifted_grasp": 1.0,
                "is_placed": 1.0,
                "is_released": 0.0,
                "is_settled": 0.0,
                "termination_reason": "time_limit",
            },
            {
                "reward": 1.0,
                "length": 50,
                "is_success": 0.0,
                "best_success_streak": 0,
                "threshold_crossed": 0.0,
                "steps_above_success_height": 0,
                "contact_stability": 0.1,
                "dual_contact_stability": 0.0,
                "max_lift_height": 0.03,
                "goal_distance_xy": 0.20,
                "best_goal_distance_xy": 0.12,
                "episode_has_grasped": 0.0,
                "episode_has_lifted_grasp": 0.0,
                "is_placed": 0.0,
                "is_released": 0.0,
                "is_settled": 0.0,
                "termination_reason": "dropped",
            },
        ],
        split="validation",
        base_seed=20_000,
        num_timesteps=25_000,
        near_success_threshold=5,
    )

    assert summary.success_rate == pytest.approx(1.0 / 3.0)
    assert summary.near_success_rate == pytest.approx(2.0 / 3.0)
    assert summary.threshold_cross_rate == pytest.approx(2.0 / 3.0)
    assert summary.mean_best_success_streak == pytest.approx(5.0)
    assert summary.mean_dual_contact_stability == pytest.approx((0.7 + 0.4 + 0.0) / 3.0)
    assert summary.mean_goal_distance_xy == pytest.approx((0.01 + 0.03 + 0.20) / 3.0)
    assert summary.grasp_rate == pytest.approx(2.0 / 3.0)
    assert summary.lifted_grasp_rate == pytest.approx(2.0 / 3.0)
    assert summary.placement_rate == pytest.approx(2.0 / 3.0)
    assert summary.release_rate == pytest.approx(1.0 / 3.0)
    assert summary.settle_rate == pytest.approx(1.0 / 3.0)
    assert summary.termination_reason_counts == {
        "dropped": 1,
        "success": 1,
        "time_limit": 1,
    }
