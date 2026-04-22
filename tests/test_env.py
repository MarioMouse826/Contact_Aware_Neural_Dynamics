from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env

from contact_aware_rl.config import EnvConfig, RewardConfig
from contact_aware_rl.env import ContactAwareGraspLiftEnv


def make_env(*, mode: str = "contact", override: str | None = None) -> ContactAwareGraspLiftEnv:
    return ContactAwareGraspLiftEnv(
        EnvConfig(
            observation_mode=mode,
            contact_override=override,
            max_episode_steps=50,
        ),
        RewardConfig(),
    )


def test_env_passes_sb3_checker() -> None:
    env = make_env()
    check_env(env, warn=True, skip_render_check=True)
    env.close()


def test_contact_extraction_no_left_right_and_both() -> None:
    env = make_env()

    env.set_manual_configuration(
        gripper_xyz=(0.0, 0.0, 0.08),
        finger_positions=(0.0, 0.0),
        object_position=(0.0, 0.0, 0.08),
    )
    assert np.array_equal(env._get_true_contact_bits(), np.array([0.0, 0.0], dtype=np.float32))

    env.set_manual_configuration(
        gripper_xyz=(0.0, 0.0, 0.08),
        finger_positions=(0.05, 0.0),
        object_position=(-0.044, 0.0, 0.08),
    )
    assert np.array_equal(env._get_true_contact_bits(), np.array([1.0, 0.0], dtype=np.float32))

    env.set_manual_configuration(
        gripper_xyz=(0.0, 0.0, 0.08),
        finger_positions=(0.0, 0.05),
        object_position=(0.044, 0.0, 0.08),
    )
    assert np.array_equal(env._get_true_contact_bits(), np.array([0.0, 1.0], dtype=np.float32))

    env.set_manual_configuration(
        gripper_xyz=(0.0, 0.0, 0.08),
        finger_positions=(0.048, 0.048),
        object_position=(0.0, 0.0, 0.08),
    )
    assert np.array_equal(env._get_true_contact_bits(), np.array([1.0, 1.0], dtype=np.float32))
    env.close()


def test_observation_shapes_and_contact_overrides() -> None:
    baseline_env = make_env(mode="baseline")
    contact_env = make_env(mode="contact")
    always_env = make_env(mode="contact", override="ones")
    zero_env = make_env(mode="contact", override="zeros")

    baseline_obs, _ = baseline_env.reset(seed=0)
    contact_obs, _ = contact_env.reset(seed=0)
    always_obs, _ = always_env.reset(seed=0)
    zero_obs, _ = zero_env.reset(seed=0)

    assert baseline_obs.shape == (26,)
    assert contact_obs.shape == (28,)
    assert np.all(always_obs[-2:] == 1.0)
    assert np.all(zero_obs[-2:] == 0.0)

    baseline_env.close()
    contact_env.close()
    always_env.close()
    zero_env.close()


def test_reward_terms_increase_with_contact_and_lift() -> None:
    env = make_env()

    env.set_manual_configuration(
        gripper_xyz=(0.0, 0.0, 0.08),
        finger_positions=(0.0, 0.0),
        object_position=(0.0, 0.0, 0.08),
    )
    no_contact = env._compute_reward_terms(
        np.zeros(4, dtype=np.float32),
        env._get_true_contact_bits(),
    )

    env.set_manual_configuration(
        gripper_xyz=(0.0, 0.0, 0.08),
        finger_positions=(0.048, 0.048),
        object_position=(0.0, 0.0, 0.08),
    )
    with_contact = env._compute_reward_terms(
        np.zeros(4, dtype=np.float32),
        env._get_true_contact_bits(),
    )

    env.set_manual_configuration(
        gripper_xyz=(0.0, 0.0, 0.15),
        finger_positions=(0.048, 0.048),
        object_position=(0.0, 0.0, 0.16),
    )
    lifted = env._compute_reward_terms(
        np.zeros(4, dtype=np.float32),
        env._get_true_contact_bits(),
    )

    assert with_contact.contact > no_contact.contact
    assert lifted.lift > with_contact.lift
    env.close()


def test_scripted_grasp_sequence_reaches_success() -> None:
    env = ContactAwareGraspLiftEnv(
        EnvConfig(observation_mode="contact", max_episode_steps=200),
        RewardConfig(),
    )
    env.reset(seed=0)

    scripted_phases = [
        np.array([0.0, 0.0, -0.8, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 0.4], dtype=np.float32),
    ]
    scripted_steps = [20, 18, 25]

    success = False
    for action, steps in zip(scripted_phases, scripted_steps):
        for _ in range(steps):
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                success = bool(info["is_success"])
                break
        if success:
            break

    assert success
    env.close()
