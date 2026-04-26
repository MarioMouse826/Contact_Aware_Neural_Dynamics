from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    embodiment: str = "cartesian_gripper"
    task: str = "pick_place_ab"
    observation_mode: str = "contact"
    contact_override: str | None = None
    max_episode_steps: int = 200
    substeps: int = 10
    action_scale_xyz: float = 0.015
    action_scale_grip: float = 0.01
    action_smoothing: float = 0.0
    transport_z_action_scale: float = 1.0
    arm_joint_delta_scales: list[float] = field(
        default_factory=lambda: [0.08, 0.08, 0.08, 0.08]
    )
    arm_control_mode: str = "joint_delta"
    arm_ik_damping: float = 1e-3
    table_height: float = 0.05
    success_height_over_table: float = 0.08
    success_hold_steps: int = 10
    termination_drop_margin: float = 0.03
    reset_object_xy_range: float = 0.035
    reset_object_yaw_range: float = math.pi / 6.0
    reset_gripper_xy_noise: float = 0.025
    reset_gripper_z_noise: float = 0.01
    reset_arm_joint_noise: float = 0.05
    pick_place_start_xy: list[float] = field(default_factory=lambda: [-0.12, -0.12])
    pick_place_goal_xy: list[float] = field(default_factory=lambda: [0.12, 0.12])
    pick_place_object_yaw: float = 0.0
    pick_place_goal_tolerance_xy: float = 0.03
    pick_place_rest_height_tolerance: float = 0.012
    pick_place_transport_clearance: float = 0.04
    pick_place_transport_height_tolerance: float = 0.015
    pick_place_transport_goal_radius: float = 0.06
    pick_place_settle_speed_threshold: float = 0.05
    grasp_alignment_threshold: float = 0.08
    pick_place_goal_hold_steps: int = 10
    initial_gripper_height: float = 0.16
    initial_finger_position: float = 0.0
    initial_arm_joint_positions: list[float] = field(
        default_factory=lambda: [-0.01, 0.46, -0.97, -1.68]
    )
    object_half_extents: list[float] = field(
        default_factory=lambda: [0.025, 0.025, 0.03]
    )
    object_mass: float = 0.03
    object_friction: list[float] = field(default_factory=lambda: [1.2, 0.05, 0.01])
    finger_friction: list[float] = field(default_factory=lambda: [1.8, 0.08, 0.015])


@dataclass
class RewardConfig:
    reach_weight: float = 1.0
    contact_weight: float = 0.25
    grasp_alignment_weight: float = 0.0
    start_stability_weight: float = 0.0
    lift_weight: float = 5.0
    transport_weight: float = 4.0
    place_weight: float = 3.0
    release_weight: float = 1.0
    settle_weight: float = 2.0
    hold_weight: float = 2.0
    success_bonus: float = 10.0
    action_penalty_weight: float = 0.01
    action_delta_penalty_weight: float = 0.02
    lift_instability_penalty_weight: float = 0.5
    lift_instability_speed_threshold: float = 0.12
    carry_height_bonus_weight: float = 0.0
    transport_vertical_speed_penalty_weight: float = 0.0
    release_ready_open_bonus_weight: float = 0.0
    release_ready_hold_penalty_weight: float = 0.0
    post_release_retreat_bonus_weight: float = 0.0
    post_release_recontact_penalty_weight: float = 0.0


@dataclass
class TrainConfig:
    seed: int = 0
    num_envs: int = 1
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    eval_freq: int = 25_000
    checkpoint_freq: int = 100_000
    early_stop_success_rate: float = 0.8
    early_stop_success_patience: int = 2
    early_stop_plateau_patience: int = 5
    early_stop_plateau_start_timesteps: int = 150_000
    device: str = "auto"


@dataclass
class EvalSplitConfig:
    episodes: int
    seed_offset: int


@dataclass
class EvalConfig:
    deterministic: bool = True
    monitor: EvalSplitConfig = field(
        default_factory=lambda: EvalSplitConfig(episodes=20, seed_offset=10_000)
    )
    validation: EvalSplitConfig = field(
        default_factory=lambda: EvalSplitConfig(episodes=100, seed_offset=20_000)
    )
    test: EvalSplitConfig = field(
        default_factory=lambda: EvalSplitConfig(episodes=100, seed_offset=30_000)
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "EvalConfig":
        payload = payload or {}
        legacy_episodes = payload.get("episodes")
        return cls(
            deterministic=payload.get("deterministic", True),
            monitor=_build_eval_split_config(
                payload=payload.get("monitor"),
                default_episodes=20,
                default_seed_offset=10_000,
                legacy_episodes=legacy_episodes,
            ),
            validation=_build_eval_split_config(
                payload=payload.get("validation"),
                default_episodes=100,
                default_seed_offset=20_000,
                legacy_episodes=legacy_episodes,
            ),
            test=_build_eval_split_config(
                payload=payload.get("test"),
                default_episodes=100,
                default_seed_offset=30_000,
                legacy_episodes=legacy_episodes,
            ),
        )


@dataclass
class LoggingConfig:
    wandb_entity: str = "contact-aware-rl"
    wandb_project: str = "contact-aware-neural-dynamics"
    wandb_mode: str = "online"
    sync_tensorboard: bool = True
    save_code: bool = True
    gradient_save_freq: int = 0
    model_save_freq: int = 0
    output_root: str = "outputs"
    wandb_tags: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def clone(self) -> "ExperimentConfig":
        return ExperimentConfig.from_dict(copy.deepcopy(self.to_dict()))

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "ExperimentConfig":
        payload = payload or {}
        return cls(
            env=EnvConfig(**payload.get("env", {})),
            reward=RewardConfig(**payload.get("reward", {})),
            train=TrainConfig(**payload.get("train", {})),
            eval=EvalConfig.from_dict(payload.get("eval", {})),
            logging=LoggingConfig(**payload.get("logging", {})),
        )


def _build_eval_split_config(
    *,
    payload: dict[str, Any] | None,
    default_episodes: int,
    default_seed_offset: int,
    legacy_episodes: int | None,
) -> EvalSplitConfig:
    payload = payload or {}
    return EvalSplitConfig(
        episodes=int(
            payload.get(
                "episodes",
                legacy_episodes if legacy_episodes is not None else default_episodes,
            )
        ),
        seed_offset=int(payload.get("seed_offset", default_seed_offset)),
    )


def _merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if (
            isinstance(value, dict)
            and key in merged
            and isinstance(merged[key], dict)
        ):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    config = ExperimentConfig()
    if path is None:
        return config

    raw_payload = yaml.safe_load(Path(path).read_text()) or {}
    merged_payload = _merge_dicts(config.to_dict(), raw_payload)
    return ExperimentConfig.from_dict(merged_payload)


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
