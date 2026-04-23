from __future__ import annotations

from pathlib import Path

from contact_aware_rl.callbacks import PeriodicEvalCallback
from contact_aware_rl.evaluation import EvaluationSummary


def _summary(
    *,
    over_goal_rate: float,
    transport_ready_rate: float,
    lifted_grasp_rate: float,
    mean_best_goal_distance_xy: float,
    mean_reward: float,
) -> EvaluationSummary:
    return EvaluationSummary(
        task="pick_place_ab",
        split="validation",
        base_seed=20_000,
        num_timesteps=200_000,
        num_episodes=20,
        mean_reward=mean_reward,
        success_rate=0.0,
        near_success_rate=0.0,
        threshold_cross_rate=0.0,
        mean_best_success_streak=0.0,
        mean_episode_length=200.0,
        mean_contact_stability=0.5,
        mean_dual_contact_stability=0.2,
        mean_max_lift_height=0.06,
        mean_goal_distance_xy=0.28,
        mean_best_goal_distance_xy=mean_best_goal_distance_xy,
        grasp_rate=0.9,
        lifted_grasp_rate=lifted_grasp_rate,
        transport_ready_rate=transport_ready_rate,
        over_goal_rate=over_goal_rate,
        placement_rate=0.0,
        release_rate=0.0,
        settle_rate=0.0,
        termination_reason_counts={"time_limit": 20},
        episodes=[],
    )


def test_pick_place_priority_prefers_later_stage_progress(tmp_path: Path) -> None:
    callback = PeriodicEvalCallback(
        monitor_env=object(),
        validation_env=object(),
        output_dir=tmp_path,
        total_timesteps=500_000,
        eval_freq=25_000,
        checkpoint_freq=100_000,
        monitor_episodes=20,
        monitor_seed=10_000,
        validation_episodes=100,
        validation_seed=20_000,
        deterministic=True,
        early_stop_success_rate=0.8,
        early_stop_success_patience=2,
        early_stop_plateau_patience=5,
        early_stop_plateau_start_timesteps=150_000,
    )

    stuck_summary = _summary(
        over_goal_rate=0.0,
        transport_ready_rate=0.0,
        lifted_grasp_rate=0.7,
        mean_best_goal_distance_xy=0.30,
        mean_reward=1.2,
    )
    progress_summary = _summary(
        over_goal_rate=0.2,
        transport_ready_rate=0.6,
        lifted_grasp_rate=0.5,
        mean_best_goal_distance_xy=0.26,
        mean_reward=0.3,
    )

    assert callback._priority(progress_summary) > callback._priority(stuck_summary)
