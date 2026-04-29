from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from contact_aware_rl.config import EnvConfig, RewardConfig
from contact_aware_rl.env import (
    ArmPinchGraspLiftEnv,
    ContactAwareGraspLiftEnv,
    TaskStatus,
    make_env,
)

LIFT_OBJECT_POS = (0.0, 0.0, 0.08)
PICK_PLACE_START_POS = (-0.12, -0.12, 0.08)
PICK_PLACE_GOAL_POS = (0.12, 0.12, 0.08)

ARM_CONTACT_POSES: dict[str, tuple[np.ndarray, tuple[float, float], np.ndarray]] = {
    "none": (
        np.array([0.2604372, 0.74606935, -1.37162827, -1.01526105], dtype=np.float64),
        (0.02267991, 0.02979928),
        np.array([0.0, 0.0], dtype=np.float32),
    ),
    "left": (
        np.array([-0.12147974, 0.69076101, -1.38235517, -1.29045246], dtype=np.float64),
        (0.00515804, 0.02500329),
        np.array([1.0, 0.0], dtype=np.float32),
    ),
    "right": (
        np.array([0.117752, 0.46185739, -1.0252404, -1.56657285], dtype=np.float64),
        (0.00741623, 0.01543839),
        np.array([0.0, 1.0], dtype=np.float32),
    ),
    "both": (
        np.array([-0.00859175, 0.45848369, -0.96914701, -1.67733621], dtype=np.float64),
        (0.01401319, 0.02270039),
        np.array([1.0, 1.0], dtype=np.float32),
    ),
}
ARM_PICK_PLACE_START_CONTACT_POSE = np.array(
    [1.1510553, 0.3717711, -1.40503214, -1.15755891],
    dtype=np.float64,
)
ARM_PICK_PLACE_START_CONTACT_FINGERS = (0.02398491, 0.00579478)
ARM_PICK_PLACE_GOAL_CONTACT_POSE = np.array(
    [-0.35463111, 0.09097982, -0.59791054, -0.44003162],
    dtype=np.float64,
)
ARM_PICK_PLACE_GOAL_CONTACT_FINGERS = (0.0206883, 0.00913122)


def create_env(
    *,
    embodiment: str = "cartesian_gripper",
    task: str | None = None,
    mode: str = "contact",
    override: str | None = None,
    max_episode_steps: int = 50,
):
    resolved_task = task or "pick_place_ab"
    return make_env(
        EnvConfig(
            embodiment=embodiment,
            task=resolved_task,
            observation_mode=mode,
            contact_override=override,
            max_episode_steps=max_episode_steps,
        ),
        RewardConfig(),
    )


def create_pick_place_env(max_episode_steps: int = 200) -> ContactAwareGraspLiftEnv:
    return ContactAwareGraspLiftEnv(
        EnvConfig(
            task="pick_place_ab",
            observation_mode="contact",
            max_episode_steps=max_episode_steps,
        ),
        RewardConfig(),
    )


def create_arm_pick_place_env(max_episode_steps: int = 200) -> ArmPinchGraspLiftEnv:
    return ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            max_episode_steps=max_episode_steps,
        ),
        RewardConfig(),
    )


def _pick_place_noop(env: ContactAwareGraspLiftEnv | ArmPinchGraspLiftEnv) -> np.ndarray:
    return np.zeros(env.action_space.shape[0], dtype=np.float32)


OBJECT_SHAPE_CASES = [
    ("sphere", {"object_shape": "sphere", "object_radius": 0.030}),
    (
        "rectangular_block",
        {
            "object_shape": "box",
            "object_half_extents": [0.035, 0.020, 0.030],
        },
    ),
    (
        "cylinder",
        {
            "object_shape": "cylinder",
            "object_radius": 0.025,
            "object_half_extents": [0.025, 0.025, 0.030],
        },
    ),
    (
        "triangular_prism",
        {
            "object_shape": "triangular_prism",
            "object_radius": 0.029,
            "object_half_extents": [0.029, 0.029, 0.030],
        },
    ),
]


def _prime_pick_place_transport(env: ContactAwareGraspLiftEnv) -> None:
    noop = _pick_place_noop(env)
    lift_height = PICK_PLACE_START_POS[2] + 0.06

    env.set_manual_configuration(
        gripper_xyz=(PICK_PLACE_START_POS[0], PICK_PLACE_START_POS[1], lift_height),
        finger_positions=(0.048, 0.048),
        object_position=(PICK_PLACE_START_POS[0], PICK_PLACE_START_POS[1], lift_height),
    )
    env.step(noop)

    env.set_manual_configuration(
        gripper_xyz=(PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], lift_height),
        finger_positions=(0.048, 0.048),
        object_position=(PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], lift_height),
    )
    env.step(noop)


def _prime_arm_pick_place_transport(env: ArmPinchGraspLiftEnv) -> None:
    noop = _pick_place_noop(env)
    lift_height = PICK_PLACE_START_POS[2] + 0.06

    env.set_manual_configuration(
        arm_joint_positions=ARM_PICK_PLACE_START_CONTACT_POSE,
        finger_positions=ARM_PICK_PLACE_START_CONTACT_FINGERS,
        object_position=PICK_PLACE_START_POS,
    )
    env.step(noop)

    env.set_manual_configuration(
        arm_joint_positions=ARM_PICK_PLACE_START_CONTACT_POSE,
        finger_positions=ARM_PICK_PLACE_START_CONTACT_FINGERS,
        object_position=(PICK_PLACE_START_POS[0], PICK_PLACE_START_POS[1], lift_height),
    )
    env.step(noop)

    env.set_manual_configuration(
        arm_joint_positions=ARM_PICK_PLACE_GOAL_CONTACT_POSE,
        finger_positions=ARM_PICK_PLACE_GOAL_CONTACT_FINGERS,
        object_position=(PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], lift_height),
    )
    env.step(noop)


@pytest.mark.parametrize(
    ("embodiment", "task"),
    [
        ("cartesian_gripper", "pick_place_ab"),
        ("cartesian_gripper", "grasp_lift"),
        ("arm_pinch", "pick_place_ab"),
        ("arm_pinch", "grasp_lift"),
    ],
)
def test_env_passes_sb3_checker(embodiment: str, task: str) -> None:
    env = create_env(embodiment=embodiment, task=task)
    check_env(env, warn=True, skip_render_check=True)
    env.close()


def test_cartesian_contact_extraction_no_left_right_and_both() -> None:
    env = ContactAwareGraspLiftEnv(
        EnvConfig(task="grasp_lift", observation_mode="contact", max_episode_steps=50),
        RewardConfig(),
    )

    env.set_manual_configuration(
        gripper_xyz=(0.0, 0.0, 0.08),
        finger_positions=(0.0, 0.0),
        object_position=LIFT_OBJECT_POS,
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
        object_position=LIFT_OBJECT_POS,
    )
    assert np.array_equal(env._get_true_contact_bits(), np.array([1.0, 1.0], dtype=np.float32))
    env.close()


def test_arm_contact_extraction_no_left_right_and_both() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="grasp_lift",
            observation_mode="contact",
            max_episode_steps=50,
        ),
        RewardConfig(),
    )

    for pose_name in ("none", "left", "right", "both"):
        arm_joint_positions, finger_positions, expected_bits = ARM_CONTACT_POSES[pose_name]
        env.set_manual_configuration(
            arm_joint_positions=arm_joint_positions,
            finger_positions=finger_positions,
            object_position=LIFT_OBJECT_POS,
        )
        assert np.array_equal(env._get_true_contact_bits(), expected_bits)

    env.close()


def test_pick_place_reset_uses_explicit_start_goal_and_home_pose() -> None:
    env = create_pick_place_env()
    observation, info = env.reset(seed=0)
    object_pos, object_quat = env._get_object_state()
    control_state, _ = env._get_control_state()
    repeat_observation, _ = env.reset(seed=0)
    repeat_object_pos, repeat_object_quat = env._get_object_state()
    varied_observation, _ = env.reset(seed=1)
    varied_object_pos, varied_object_quat = env._get_object_state()

    assert np.allclose(object_pos, repeat_object_pos)
    assert np.allclose(object_quat, repeat_object_quat)
    assert np.allclose(observation, repeat_observation)
    assert not np.allclose(object_pos[:2], varied_object_pos[:2])
    assert not np.allclose(object_quat, varied_object_quat)
    assert abs(object_pos[0] - PICK_PLACE_START_POS[0]) <= env.env_config.reset_object_xy_range
    assert abs(object_pos[1] - PICK_PLACE_START_POS[1]) <= env.env_config.reset_object_xy_range
    assert object_pos[2] == pytest.approx(PICK_PLACE_START_POS[2])
    assert np.allclose(control_state[:3], np.array([0.0, 0.0, 0.16], dtype=np.float64))
    assert observation.shape == (34,)
    assert np.allclose(observation[-8:-5], np.array(PICK_PLACE_GOAL_POS, dtype=np.float32))
    assert np.allclose(observation[-5:-2], np.array(PICK_PLACE_GOAL_POS, dtype=np.float32) - object_pos)
    assert info["task"] == "pick_place_ab"
    assert info["goal_distance_xy"] == pytest.approx(
        np.linalg.norm(np.array(PICK_PLACE_GOAL_POS[:2], dtype=np.float64) - object_pos[:2])
    )
    env.close()


def test_arm_pick_place_reset_uses_explicit_start_goal_and_home_pose() -> None:
    env = create_arm_pick_place_env()
    observation, info = env.reset(seed=0)
    object_pos, object_quat = env._get_object_state()
    control_state, _ = env._get_control_state()
    repeat_observation, _ = env.reset(seed=0)
    repeat_object_pos, repeat_object_quat = env._get_object_state()
    repeat_control_state, _ = env._get_control_state()
    varied_observation, _ = env.reset(seed=1)
    varied_object_pos, varied_object_quat = env._get_object_state()
    varied_control_state, _ = env._get_control_state()

    assert np.allclose(object_pos, repeat_object_pos)
    assert np.allclose(object_quat, repeat_object_quat)
    assert np.allclose(control_state, repeat_control_state)
    assert np.allclose(observation, repeat_observation)
    assert not np.allclose(object_pos[:2], varied_object_pos[:2])
    assert not np.allclose(object_quat, varied_object_quat)
    assert not np.allclose(control_state[:4], varied_control_state[:4])
    assert abs(object_pos[0] - PICK_PLACE_START_POS[0]) <= env.env_config.reset_object_xy_range
    assert abs(object_pos[1] - PICK_PLACE_START_POS[1]) <= env.env_config.reset_object_xy_range
    assert object_pos[2] == pytest.approx(PICK_PLACE_START_POS[2])
    assert np.allclose(control_state[4:], np.array([0.0, 0.0], dtype=np.float64))
    assert observation.shape == (36,)
    assert np.allclose(observation[-8:-5], np.array(PICK_PLACE_GOAL_POS, dtype=np.float32))
    assert info["task"] == "pick_place_ab"
    env.close()


def test_default_object_shape_is_original_box() -> None:
    config = EnvConfig()
    assert config.object_shape == "box"
    assert config.object_radius is None
    assert config.object_half_extents == [0.025, 0.025, 0.03]


@pytest.mark.parametrize(("object_name", "shape_overrides"), OBJECT_SHAPE_CASES)
@pytest.mark.parametrize(
    ("embodiment", "expected_obs_dim", "expected_action_dim"),
    [
        ("cartesian_gripper", 34, 4),
        ("arm_pinch", 36, 5),
    ],
)
def test_pick_place_supported_object_shapes_compile_and_reset(
    object_name: str,
    shape_overrides: dict[str, object],
    embodiment: str,
    expected_obs_dim: int,
    expected_action_dim: int,
) -> None:
    del object_name
    config = EnvConfig(
        embodiment=embodiment,
        task="pick_place_ab",
        observation_mode="contact",
        max_episode_steps=5,
        **shape_overrides,
    )
    env = make_env(config, RewardConfig())
    observation, info = env.reset(seed=0)
    object_pos, _ = env._get_object_state()

    assert observation.shape == (expected_obs_dim,)
    assert env.action_space.shape == (expected_action_dim,)
    assert object_pos[2] == pytest.approx(0.08)
    assert info["object_height"] == pytest.approx(0.03)
    env.close()


def test_cartesian_observation_shapes_and_contact_overrides() -> None:
    baseline_env = create_env(embodiment="cartesian_gripper", mode="baseline")
    contact_env = create_env(embodiment="cartesian_gripper", mode="contact")
    always_env = create_env(
        embodiment="cartesian_gripper",
        mode="contact",
        override="ones",
    )
    zero_env = create_env(
        embodiment="cartesian_gripper",
        mode="contact",
        override="zeros",
    )
    legacy_env = create_env(
        embodiment="cartesian_gripper",
        task="grasp_lift",
        mode="contact",
    )

    baseline_obs, _ = baseline_env.reset(seed=0)
    contact_obs, _ = contact_env.reset(seed=0)
    always_obs, _ = always_env.reset(seed=0)
    zero_obs, _ = zero_env.reset(seed=0)
    legacy_obs, _ = legacy_env.reset(seed=0)

    assert baseline_obs.shape == (32,)
    assert contact_obs.shape == (34,)
    assert np.all(always_obs[-2:] == 1.0)
    assert np.all(zero_obs[-2:] == 0.0)
    assert legacy_obs.shape == (28,)

    baseline_env.close()
    contact_env.close()
    always_env.close()
    zero_env.close()
    legacy_env.close()


def test_arm_observation_shapes_and_contact_overrides() -> None:
    baseline_env = create_env(embodiment="arm_pinch", mode="baseline")
    contact_env = create_env(embodiment="arm_pinch", mode="contact")
    ee_contact_env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            arm_control_mode="ee_delta",
        ),
        RewardConfig(),
    )
    zero_env = create_env(
        embodiment="arm_pinch",
        mode="contact",
        override="zeros",
    )
    legacy_env = create_env(
        embodiment="arm_pinch",
        task="grasp_lift",
        mode="contact",
    )

    baseline_obs, _ = baseline_env.reset(seed=0)
    contact_obs, _ = contact_env.reset(seed=0)
    ee_contact_obs, _ = ee_contact_env.reset(seed=0)
    zero_obs, _ = zero_env.reset(seed=0)
    legacy_obs, _ = legacy_env.reset(seed=0)

    assert baseline_obs.shape == (34,)
    assert contact_obs.shape == (36,)
    assert ee_contact_obs.shape == (36,)
    assert contact_env.action_space.shape == (5,)
    assert ee_contact_env.action_space.shape == (4,)
    assert np.all(zero_obs[-2:] == 0.0)
    assert legacy_obs.shape == (30,)

    baseline_env.close()
    contact_env.close()
    ee_contact_env.close()
    zero_env.close()
    legacy_env.close()


def test_arm_env_rejects_always_contact_override() -> None:
    with pytest.raises(ValueError, match="always-contact"):
        create_env(embodiment="arm_pinch", task="grasp_lift", mode="contact", override="ones")


def test_arm_ee_delta_action_moves_fingertip_center_in_requested_direction() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            arm_control_mode="ee_delta",
            reset_arm_joint_noise=0.0,
            substeps=20,
        ),
        RewardConfig(),
    )
    env.reset(seed=0)
    before = 0.5 * sum(env._get_fingertip_positions())

    env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    after = 0.5 * sum(env._get_fingertip_positions())
    assert after[0] > before[0]
    env.close()


def test_arm_ee_delta_action_respects_joint_clamps_and_ranges() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            arm_control_mode="ee_delta",
            arm_joint_delta_scales=[0.01, 0.02, 0.03, 0.04],
            reset_arm_joint_noise=0.0,
        ),
        RewardConfig(),
    )
    env.reset(seed=0)
    previous_target = env._target_ctrl.copy()

    env._set_targets_from_action(np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32))

    target_delta = np.abs(env._target_ctrl[:4] - previous_target[:4])
    assert np.all(target_delta <= np.array([0.01, 0.02, 0.03, 0.04]) + 1e-9)
    assert np.all(env._target_ctrl >= env._ctrl_low)
    assert np.all(env._target_ctrl <= env._ctrl_high)
    env.close()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("transport_z_action_scale", 0.0, "transport_z_action_scale"),
        ("pick_place_transport_height_tolerance", 0.0, "transport_height_tolerance"),
        ("pick_place_transport_goal_radius", 0.0, "transport_goal_radius"),
    ],
)
def test_pick_place_clean_release_config_validation(
    field: str,
    value: float,
    message: str,
) -> None:
    kwargs = {
        "embodiment": "arm_pinch",
        "task": "pick_place_ab",
        "observation_mode": "contact",
        field: value,
    }

    with pytest.raises(ValueError, match=message):
        ArmPinchGraspLiftEnv(EnvConfig(**kwargs), RewardConfig())


def test_arm_ee_transport_constraints_reduce_vertical_command() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            arm_control_mode="ee_delta",
            transport_z_action_scale=0.5,
        ),
        RewardConfig(),
    )
    env.reset(seed=0)
    env._episode_has_lifted_for_transport = True
    status = TaskStatus(
        lift_clearance=0.06,
        goal_distance_xy=0.20,
        goal_height_error=0.06,
        has_any_contact=True,
    )

    constrained = env._apply_transport_phase_action_constraints(
        np.array([0.2, -0.1, 1.0, 0.3], dtype=np.float32),
        status,
    )

    assert env._transport_phase_active(status)
    assert np.isclose(constrained[2], 0.5)
    assert np.isclose(constrained[0], 0.2)
    env.close()


def test_arm_joint_transport_constraints_do_not_scale_joint_action() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            arm_control_mode="joint_delta",
            transport_z_action_scale=0.5,
        ),
        RewardConfig(),
    )
    env.reset(seed=0)
    env._episode_has_lifted_for_transport = True
    status = TaskStatus(
        lift_clearance=0.06,
        goal_distance_xy=0.20,
        goal_height_error=0.06,
        has_any_contact=True,
    )

    action = np.array([0.2, -0.1, 1.0, 0.3, -0.4], dtype=np.float32)
    constrained = env._apply_transport_phase_action_constraints(action, status)

    assert env._transport_phase_active(status)
    assert np.allclose(constrained, action)
    env.close()


def test_cartesian_reward_potential_increases_with_two_finger_contact_and_lift() -> None:
    env = create_env(embodiment="cartesian_gripper", task="grasp_lift")

    env.set_manual_configuration(
        gripper_xyz=(0.0, 0.0, 0.08),
        finger_positions=(0.0, 0.0),
        object_position=LIFT_OBJECT_POS,
    )
    no_contact = env._compute_potential_terms(env._get_true_contact_bits())

    env.set_manual_configuration(
        gripper_xyz=(0.0, 0.0, 0.08),
        finger_positions=(0.05, 0.0),
        object_position=(-0.044, 0.0, 0.08),
    )
    single_contact = env._compute_potential_terms(env._get_true_contact_bits())

    env.set_manual_configuration(
        gripper_xyz=(0.0, 0.0, 0.08),
        finger_positions=(0.048, 0.048),
        object_position=LIFT_OBJECT_POS,
    )
    with_contact = env._compute_potential_terms(env._get_true_contact_bits())

    env.set_manual_configuration(
        gripper_xyz=(0.0, 0.0, 0.15),
        finger_positions=(0.048, 0.048),
        object_position=(0.0, 0.0, 0.16),
    )
    lifted = env._compute_potential_terms(env._get_true_contact_bits())

    assert single_contact.contact > no_contact.contact
    assert with_contact.contact > no_contact.contact
    assert with_contact.contact > single_contact.contact
    assert lifted.lift > with_contact.lift
    env.close()


def test_pick_place_status_distinguishes_secure_grasp_from_lifted_grasp() -> None:
    env = create_pick_place_env()
    env.reset(seed=0)

    env.set_manual_configuration(
        gripper_xyz=PICK_PLACE_START_POS,
        finger_positions=(0.048, 0.048),
        object_position=PICK_PLACE_START_POS,
    )
    grasp_status = env._build_task_status(env._get_true_contact_bits())
    assert grasp_status.is_grasped
    assert not grasp_status.is_lifted_grasp

    lift_height = PICK_PLACE_GOAL_POS[2] + 0.06
    env.set_manual_configuration(
        gripper_xyz=(PICK_PLACE_START_POS[0], PICK_PLACE_START_POS[1], lift_height),
        finger_positions=(0.048, 0.048),
        object_position=(PICK_PLACE_START_POS[0], PICK_PLACE_START_POS[1], lift_height),
    )
    lifted_status = env._build_task_status(env._get_true_contact_bits())
    assert lifted_status.is_grasped
    assert lifted_status.is_lifted_grasp
    env.close()


def test_pick_place_transport_reward_requires_secure_grasp() -> None:
    env = create_pick_place_env()
    env.reset(seed=0)

    lift_height = PICK_PLACE_GOAL_POS[2] + 0.06
    env.set_manual_configuration(
        gripper_xyz=(PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], lift_height),
        finger_positions=(0.0, 0.0),
        object_position=(PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], lift_height),
    )
    no_grasp_status = env._build_task_status(env._get_true_contact_bits())
    no_grasp_potential = env._compute_potential_terms(env._get_true_contact_bits(), no_grasp_status)
    assert no_grasp_potential.transport == 0.0

    env.reset(seed=0)
    _prime_pick_place_transport(env)
    valid_status = env._build_task_status(env._get_true_contact_bits())
    valid_potential = env._compute_potential_terms(env._get_true_contact_bits(), valid_status)
    assert valid_potential.transport > 0.0
    env.close()


def test_pick_place_transport_reward_requires_lift_before_xy_transport() -> None:
    env = create_pick_place_env()
    env.reset(seed=0)

    env.set_manual_configuration(
        gripper_xyz=PICK_PLACE_GOAL_POS,
        finger_positions=(0.048, 0.048),
        object_position=PICK_PLACE_GOAL_POS,
    )
    grounded_status = env._build_task_status(env._get_true_contact_bits())
    grounded_potential = env._compute_potential_terms(
        env._get_true_contact_bits(),
        grounded_status,
    )

    lift_height = PICK_PLACE_GOAL_POS[2] + 0.06
    env.set_manual_configuration(
        gripper_xyz=(PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], lift_height),
        finger_positions=(0.048, 0.048),
        object_position=(PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], lift_height),
    )
    lifted_status = env._build_task_status(env._get_true_contact_bits())
    lifted_potential = env._compute_potential_terms(
        env._get_true_contact_bits(),
        lifted_status,
    )

    assert grounded_status.is_grasped
    assert not grounded_status.is_lifted_grasp
    assert grounded_potential.transport == 0.0
    assert grounded_potential.place == 0.0
    assert lifted_status.is_lifted_grasp
    assert lifted_potential.transport > 0.0
    env.close()


def test_pick_place_contact_and_alignment_decay_after_lift() -> None:
    env = ContactAwareGraspLiftEnv(
        EnvConfig(
            task="pick_place_ab",
            observation_mode="contact",
            max_episode_steps=200,
        ),
        RewardConfig(contact_weight=0.25, grasp_alignment_weight=1.0),
    )
    env.reset(seed=0)

    env.set_manual_configuration(
        gripper_xyz=PICK_PLACE_START_POS,
        finger_positions=(0.048, 0.048),
        object_position=PICK_PLACE_START_POS,
    )
    grounded_status = env._build_task_status(env._get_true_contact_bits())
    grounded_potential = env._compute_potential_terms(
        env._get_true_contact_bits(),
        grounded_status,
    )

    env.set_manual_configuration(
        gripper_xyz=(
            PICK_PLACE_START_POS[0],
            PICK_PLACE_START_POS[1],
            PICK_PLACE_START_POS[2] + 0.06,
        ),
        finger_positions=(0.048, 0.048),
        object_position=(
            PICK_PLACE_START_POS[0],
            PICK_PLACE_START_POS[1],
            PICK_PLACE_START_POS[2] + 0.06,
        ),
    )
    lifted_status = env._build_task_status(env._get_true_contact_bits())
    lifted_potential = env._compute_potential_terms(
        env._get_true_contact_bits(),
        lifted_status,
    )

    assert grounded_status.is_grasped
    assert lifted_status.is_lifted_grasp
    assert lifted_potential.contact < grounded_potential.contact
    assert lifted_potential.grasp_alignment < grounded_potential.grasp_alignment
    env.close()


def test_pick_place_transport_reward_grows_nearer_goal() -> None:
    env = create_pick_place_env()
    env.reset(seed=0)
    lift_height = PICK_PLACE_START_POS[2] + 0.06

    env.set_manual_configuration(
        gripper_xyz=(PICK_PLACE_START_POS[0], PICK_PLACE_START_POS[1], lift_height),
        finger_positions=(0.048, 0.048),
        object_position=(PICK_PLACE_START_POS[0], PICK_PLACE_START_POS[1], lift_height),
    )
    start_status = env._build_task_status(env._get_true_contact_bits())
    start_potential = env._compute_potential_terms(env._get_true_contact_bits(), start_status)

    env._episode_has_lifted_for_transport = True
    env.set_manual_configuration(
        gripper_xyz=(PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], lift_height),
        finger_positions=(0.048, 0.048),
        object_position=(PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], lift_height),
    )
    goal_status = env._build_task_status(env._get_true_contact_bits())
    goal_potential = env._compute_potential_terms(env._get_true_contact_bits(), goal_status)

    assert start_status.is_lifted_grasp
    assert goal_status.is_lifted_grasp
    assert goal_potential.transport > start_potential.transport
    assert (goal_potential.lift + goal_potential.transport + goal_potential.place) > (
        start_potential.lift + start_potential.transport + start_potential.place
    )
    env.close()


def test_pick_place_place_shaping_activates_before_strict_xy_placement() -> None:
    env = create_pick_place_env()
    env.reset(seed=0)
    xy_offset = 2.0 * env.env_config.pick_place_goal_tolerance_xy
    near_goal_xy = (
        PICK_PLACE_GOAL_POS[0] - xy_offset,
        PICK_PLACE_GOAL_POS[1],
        PICK_PLACE_GOAL_POS[2],
    )
    env._episode_has_lifted_for_transport = True
    env.set_manual_configuration(
        gripper_xyz=near_goal_xy,
        finger_positions=(0.048, 0.048),
        object_position=near_goal_xy,
    )

    status = env._build_task_status(env._get_true_contact_bits())
    potential = env._compute_potential_terms(env._get_true_contact_bits(), status)

    assert status.is_grasped
    assert not status.is_over_goal
    assert not status.is_placed
    assert potential.place > 0.0
    env.close()


def test_pick_place_place_shaping_grows_while_descending_over_goal() -> None:
    env = create_pick_place_env()
    env.reset(seed=0)
    env._episode_has_lifted_for_transport = True

    high_goal_pos = (
        PICK_PLACE_GOAL_POS[0],
        PICK_PLACE_GOAL_POS[1],
        PICK_PLACE_GOAL_POS[2] + 0.06,
    )
    low_goal_pos = (
        PICK_PLACE_GOAL_POS[0],
        PICK_PLACE_GOAL_POS[1],
        PICK_PLACE_GOAL_POS[2] + 0.02,
    )

    env.set_manual_configuration(
        gripper_xyz=high_goal_pos,
        finger_positions=(0.048, 0.048),
        object_position=high_goal_pos,
    )
    high_status = env._build_task_status(env._get_true_contact_bits())
    high_potential = env._compute_potential_terms(env._get_true_contact_bits(), high_status)

    env.set_manual_configuration(
        gripper_xyz=low_goal_pos,
        finger_positions=(0.048, 0.048),
        object_position=low_goal_pos,
    )
    low_status = env._build_task_status(env._get_true_contact_bits())
    low_potential = env._compute_potential_terms(env._get_true_contact_bits(), low_status)

    assert high_status.is_grasped
    assert low_status.is_grasped
    assert high_status.is_over_goal
    assert low_status.is_over_goal
    assert not high_status.is_placed
    assert not low_status.is_placed
    assert low_potential.place > high_potential.place
    assert (
        low_potential.lift + low_potential.transport + low_potential.place
        > high_potential.lift + high_potential.transport + high_potential.place
    )
    env.close()


def test_action_delta_penalty_discourages_jitter() -> None:
    env = create_pick_place_env()
    env.reset(seed=0)
    bits = env._get_true_contact_bits()
    status = env._build_task_status(bits)
    potential = env._compute_potential_terms(bits, status)
    previous_action = -np.ones(env.action_space.shape[0], dtype=np.float32)
    action = np.ones(env.action_space.shape[0], dtype=np.float32)

    reward_terms = env._compute_reward_terms(
        action,
        potential,
        potential,
        success=False,
        task_status=status,
        previous_action=previous_action,
    )

    assert reward_terms.action_delta_penalty > 0.0
    assert reward_terms.total < 0.0
    env.close()


def test_arm_pick_place_penalizes_pregrasp_object_displacement() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            max_episode_steps=200,
        ),
        RewardConfig(
            contact_weight=0.25,
            grasp_alignment_weight=1.0,
            start_stability_weight=1.0,
        ),
    )

    env.set_manual_configuration(
        arm_joint_positions=ARM_CONTACT_POSES["none"][0],
        finger_positions=ARM_CONTACT_POSES["none"][1],
        object_position=PICK_PLACE_START_POS,
    )
    start_potential = env._compute_potential_terms(env._get_true_contact_bits())

    displaced_object_pos = (
        PICK_PLACE_START_POS[0] - 0.08,
        PICK_PLACE_START_POS[1] - 0.08,
        PICK_PLACE_START_POS[2],
    )
    env.set_manual_configuration(
        arm_joint_positions=ARM_CONTACT_POSES["none"][0],
        finger_positions=ARM_CONTACT_POSES["none"][1],
        object_position=displaced_object_pos,
    )
    displaced_potential = env._compute_potential_terms(env._get_true_contact_bits())
    reward_terms = env._compute_reward_terms(
        np.zeros(env.action_space.shape[0], dtype=np.float32),
        start_potential,
        displaced_potential,
        success=False,
    )

    assert displaced_potential.start_stability < start_potential.start_stability
    assert reward_terms.start_stability < 0.0
    assert reward_terms.total < 0.0
    env.close()


def test_arm_pick_place_transport_potential_grows_after_lift_latch() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            max_episode_steps=250,
        ),
        RewardConfig(transport_weight=10.0),
    )
    env.reset(seed=0)
    env._episode_has_lifted_for_transport = True
    lifted_z = PICK_PLACE_START_POS[2] + 0.03
    arm_joint_positions = np.asarray(env.env_config.initial_arm_joint_positions, dtype=np.float64)
    true_bits = np.array([1.0, 0.0], dtype=np.float32)

    env.set_manual_configuration(
        arm_joint_positions=arm_joint_positions,
        finger_positions=(0.0, 0.0),
        object_position=(PICK_PLACE_START_POS[0], PICK_PLACE_START_POS[1], lifted_z),
    )
    start_goal_distance = float(
        np.linalg.norm(np.asarray(PICK_PLACE_START_POS[:2]) - np.asarray(PICK_PLACE_GOAL_POS[:2]))
    )
    start_status = TaskStatus(
        has_any_contact=True,
        lift_clearance=lifted_z - PICK_PLACE_START_POS[2],
        goal_distance_xy=start_goal_distance,
        goal_height_error=lifted_z - PICK_PLACE_START_POS[2],
    )
    start_potential = env._compute_potential_terms(true_bits, start_status)

    halfway_xy = 0.5 * (np.asarray(PICK_PLACE_START_POS[:2]) + np.asarray(PICK_PLACE_GOAL_POS[:2]))
    env.set_manual_configuration(
        arm_joint_positions=arm_joint_positions,
        finger_positions=(0.0, 0.0),
        object_position=(float(halfway_xy[0]), float(halfway_xy[1]), lifted_z),
    )
    midway_goal_distance = float(np.linalg.norm(halfway_xy - np.asarray(PICK_PLACE_GOAL_POS[:2])))
    midway_status = TaskStatus(
        has_any_contact=True,
        lift_clearance=lifted_z - PICK_PLACE_START_POS[2],
        goal_distance_xy=midway_goal_distance,
        goal_height_error=lifted_z - PICK_PLACE_START_POS[2],
    )
    midway_potential = env._compute_potential_terms(true_bits, midway_status)

    assert not start_status.is_grasped
    assert not midway_status.is_grasped
    assert midway_potential.transport > start_potential.transport
    assert midway_potential.transport == pytest.approx(5.0, abs=0.1)
    env.close()


def test_arm_transport_reward_prefers_stable_carry_height_before_goal() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            pick_place_transport_height_tolerance=0.015,
            pick_place_transport_goal_radius=0.06,
        ),
        RewardConfig(transport_weight=10.0, carry_height_bonus_weight=0.8),
    )
    env.reset(seed=0)
    env._episode_has_lifted_for_transport = True
    goal_distance = float(
        np.linalg.norm(np.asarray(PICK_PLACE_START_POS[:2]) - np.asarray(PICK_PLACE_GOAL_POS[:2]))
    )
    target_clearance = (
        env.env_config.pick_place_transport_clearance
        + env.env_config.pick_place_transport_height_tolerance
    )
    true_bits = np.array([1.0, 0.0], dtype=np.float32)

    target_status = TaskStatus(
        has_any_contact=True,
        lift_clearance=target_clearance,
        goal_distance_xy=goal_distance,
        goal_height_error=target_clearance,
    )
    target_potential = env._compute_potential_terms(true_bits, target_status)

    high_status = TaskStatus(
        has_any_contact=True,
        lift_clearance=target_clearance + 0.04,
        goal_distance_xy=goal_distance,
        goal_height_error=target_clearance + 0.04,
    )
    high_potential = env._compute_potential_terms(true_bits, high_status)

    assert env._transport_phase_active(target_status)
    assert target_potential.transport > high_potential.transport
    env.close()


def test_arm_pick_place_carry_stage_requires_current_carry_state_after_lift_latch() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            max_episode_steps=250,
        ),
        RewardConfig(lift_weight=6.0, transport_weight=10.0),
    )
    env.reset(seed=0)
    env._episode_has_lifted_for_transport = True
    object_position = (
        PICK_PLACE_GOAL_POS[0],
        PICK_PLACE_GOAL_POS[1],
        PICK_PLACE_GOAL_POS[2] + 0.025,
    )

    env.set_manual_configuration(
        arm_joint_positions=np.asarray(env.env_config.initial_arm_joint_positions, dtype=np.float64),
        finger_positions=(0.0, 0.0),
        object_position=object_position,
    )
    no_contact_status = TaskStatus(
        lift_clearance=object_position[2] - PICK_PLACE_GOAL_POS[2],
        goal_distance_xy=0.0,
        goal_height_error=object_position[2] - PICK_PLACE_GOAL_POS[2],
    )
    no_contact_potential = env._compute_potential_terms(
        np.array([0.0, 0.0], dtype=np.float32),
        no_contact_status,
    )
    one_contact_status = TaskStatus(
        has_any_contact=True,
        lift_clearance=object_position[2] - PICK_PLACE_GOAL_POS[2],
        goal_distance_xy=0.0,
        goal_height_error=object_position[2] - PICK_PLACE_GOAL_POS[2],
    )
    one_contact_potential = env._compute_potential_terms(
        np.array([1.0, 0.0], dtype=np.float32),
        one_contact_status,
    )

    assert not no_contact_status.is_grasped
    assert no_contact_status.lift_clearance < env.env_config.pick_place_transport_clearance
    assert no_contact_potential.lift < 6.0
    assert no_contact_potential.transport == 0.0
    assert one_contact_potential.lift == pytest.approx(6.0)
    assert one_contact_potential.transport == pytest.approx(10.0)
    env.close()


def test_arm_transport_ready_latch_requires_current_grasp() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            max_episode_steps=250,
        ),
        RewardConfig(),
    )
    env.reset(seed=0)
    env._episode_has_grasped = True
    object_pos = np.array(
        [
            PICK_PLACE_START_POS[0],
            PICK_PLACE_START_POS[1],
            PICK_PLACE_START_POS[2] + 0.06,
        ],
        dtype=np.float64,
    )
    status = TaskStatus(
        lift_clearance=0.06,
        goal_distance_xy=float(
            np.linalg.norm(np.asarray(PICK_PLACE_START_POS[:2]) - np.asarray(PICK_PLACE_GOAL_POS[:2]))
        ),
        goal_height_error=0.06,
        is_grasped=False,
    )

    env._update_success_tracking(object_pos=object_pos, task_status=status)

    assert not env._episode_has_lifted_for_transport
    env.close()


def test_arm_start_stability_remains_until_transport_ready() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            max_episode_steps=250,
        ),
        RewardConfig(start_stability_weight=1.5),
    )
    env.reset(seed=0)
    env._episode_has_lifted_grasp = True
    displaced_pos = (
        PICK_PLACE_START_POS[0] + 0.08,
        PICK_PLACE_START_POS[1],
        PICK_PLACE_START_POS[2] + 0.02,
    )
    env.set_manual_configuration(
        arm_joint_positions=np.asarray(env.env_config.initial_arm_joint_positions, dtype=np.float64),
        finger_positions=(0.0, 0.0),
        object_position=displaced_pos,
    )
    status = TaskStatus(
        lift_clearance=0.02,
        goal_distance_xy=float(
            np.linalg.norm(np.asarray(displaced_pos[:2]) - np.asarray(PICK_PLACE_GOAL_POS[:2]))
        ),
        goal_height_error=0.02,
    )
    pre_transport_potential = env._compute_potential_terms(
        np.array([0.0, 0.0], dtype=np.float32),
        status,
    )

    env._episode_has_lifted_for_transport = True
    transport_ready_potential = env._compute_potential_terms(
        np.array([0.0, 0.0], dtype=np.float32),
        status,
    )

    assert pre_transport_potential.start_stability < 0.0
    assert transport_ready_potential.start_stability == 0.0
    env.close()


def test_contact_stability_tracks_any_and_dual_contact_separately() -> None:
    env = create_pick_place_env(max_episode_steps=10)
    env.reset(seed=0)

    env._episode_steps = 2
    env._episode_any_contact_steps = 2
    env._episode_dual_contact_steps = 1
    bits = np.array([1.0, 1.0], dtype=np.float32)
    status = env._build_task_status(bits)
    reward_terms = env._compute_reward_terms(
        np.zeros(env.action_space.shape[0], dtype=np.float32),
        env._previous_potential,
        env._previous_potential,
        success=False,
    )
    info = env._build_info(
        true_contact_bits=bits,
        observed_contact_bits=bits,
        task_status=status,
        reward_terms=reward_terms,
        success=False,
    )

    assert info["contact_stability"] == pytest.approx(1.0)
    assert info["dual_contact_stability"] == pytest.approx(0.5)
    env.close()


def test_pick_place_info_tracks_transport_ready_and_over_goal_flags() -> None:
    env = create_pick_place_env()
    env.reset(seed=0)
    _prime_pick_place_transport(env)
    bits = env._get_true_contact_bits()
    status = env._build_task_status(bits)
    reward_terms = env._compute_reward_terms(
        np.zeros(env.action_space.shape[0], dtype=np.float32),
        env._previous_potential,
        env._previous_potential,
        success=False,
    )
    info = env._build_info(
        true_contact_bits=bits,
        observed_contact_bits=bits,
        task_status=status,
        reward_terms=reward_terms,
        success=False,
    )

    assert info["episode_has_lifted_for_transport"] == 1.0
    assert info["episode_has_over_goal"] == 1.0
    env.close()


def test_pick_place_goal_does_not_succeed_while_still_holding_cube() -> None:
    env = create_pick_place_env()
    env.reset(seed=0)
    _prime_pick_place_transport(env)

    env.set_manual_configuration(
        gripper_xyz=(PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], 0.09),
        finger_positions=(0.048, 0.048),
        object_position=PICK_PLACE_GOAL_POS,
    )
    _, _, terminated, truncated, info = env.step(_pick_place_noop(env))

    assert not terminated
    assert not truncated
    assert info["is_success"] == 0.0
    assert info["is_placed"] == 1.0
    assert info["is_released"] == 0.0
    assert info["is_settled"] == 0.0
    env.close()


def test_pick_place_release_progress_outweighs_holding_at_goal() -> None:
    env = create_pick_place_env()
    env.reset(seed=0)
    _prime_pick_place_transport(env)

    held_goal_pos = (PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], 0.10)
    env.set_manual_configuration(
        gripper_xyz=held_goal_pos,
        finger_positions=(0.048, 0.048),
        object_position=PICK_PLACE_GOAL_POS,
    )
    held_status = env._build_task_status(env._get_true_contact_bits())
    held_potential = env._compute_potential_terms(env._get_true_contact_bits(), held_status)

    env._episode_has_lifted_for_transport = True
    env._episode_has_over_goal = True
    env._episode_has_placed = True
    released_goal_pos = (PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], 0.16)
    env.set_manual_configuration(
        gripper_xyz=released_goal_pos,
        finger_positions=(0.0, 0.0),
        object_position=PICK_PLACE_GOAL_POS,
    )
    released_status = env._build_task_status(env._get_true_contact_bits())
    released_potential = env._compute_potential_terms(
        env._get_true_contact_bits(),
        released_status,
    )

    assert held_status.is_placed
    assert not held_status.is_released
    assert released_status.is_released
    assert released_potential.release > held_potential.release
    assert released_potential.total > held_potential.total
    env.close()


def test_pick_place_release_corridor_rewards_opening_before_full_detach() -> None:
    env = create_pick_place_env()
    env.reset(seed=0)
    _prime_pick_place_transport(env)

    held_goal_pos = (PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], 0.10)
    env.set_manual_configuration(
        gripper_xyz=held_goal_pos,
        finger_positions=(0.048, 0.048),
        object_position=PICK_PLACE_GOAL_POS,
    )
    held_status = env._build_task_status(env._get_true_contact_bits())
    held_potential = env._compute_potential_terms(env._get_true_contact_bits(), held_status)

    env._episode_has_lifted_for_transport = True
    env._episode_has_over_goal = True
    env.set_manual_configuration(
        gripper_xyz=held_goal_pos,
        finger_positions=(0.0, 0.0),
        object_position=PICK_PLACE_GOAL_POS,
    )
    opened_status = env._build_task_status(env._get_true_contact_bits())
    opened_potential = env._compute_potential_terms(
        env._get_true_contact_bits(),
        opened_status,
    )

    assert held_status.is_placed
    assert opened_status.is_placed
    assert opened_potential.release > held_potential.release
    env.close()


def test_arm_release_ready_zone_rewards_opening_before_full_release() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            max_episode_steps=250,
        ),
        RewardConfig(
            release_weight=2.0,
            release_ready_open_bonus_weight=0.25,
            release_ready_hold_penalty_weight=0.08,
        ),
    )
    env.reset(seed=0)
    env._episode_has_lifted_for_transport = True
    env._episode_has_over_goal = True
    status = TaskStatus(
        has_any_contact=True,
        has_dual_contact=True,
        lift_clearance=0.0,
        goal_distance_xy=0.0,
        goal_height_error=0.0,
        object_speed=0.0,
        is_over_goal=True,
        is_placed=True,
    )
    true_bits = np.array([1.0, 1.0], dtype=np.float32)

    env.set_manual_configuration(
        arm_joint_positions=np.asarray(env.env_config.initial_arm_joint_positions, dtype=np.float64),
        finger_positions=(0.03, 0.03),
        object_position=PICK_PLACE_GOAL_POS,
    )
    held_potential = env._compute_potential_terms(true_bits, status)

    env.set_manual_configuration(
        arm_joint_positions=np.asarray(env.env_config.initial_arm_joint_positions, dtype=np.float64),
        finger_positions=(0.0, 0.0),
        object_position=PICK_PLACE_GOAL_POS,
    )
    opening_potential = env._compute_potential_terms(true_bits, status)

    assert opening_potential.release > held_potential.release
    env.close()


def test_pick_place_transport_and_place_persist_after_release() -> None:
    env = create_pick_place_env()
    env.reset(seed=0)
    _prime_pick_place_transport(env)

    env._episode_has_lifted_for_transport = True
    env._episode_has_over_goal = True
    env._episode_has_placed = True
    released_goal_pos = (PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], 0.16)
    env.set_manual_configuration(
        gripper_xyz=released_goal_pos,
        finger_positions=(0.0, 0.0),
        object_position=PICK_PLACE_GOAL_POS,
    )
    released_status = env._build_task_status(env._get_true_contact_bits())
    released_potential = env._compute_potential_terms(
        env._get_true_contact_bits(),
        released_status,
    )

    assert released_status.is_released
    assert released_potential.transport > 0.0
    assert released_potential.place > 0.0
    env.close()


def test_pick_place_release_and_settle_sequence_reaches_success() -> None:
    env = create_pick_place_env(max_episode_steps=250)
    env.reset(seed=0)
    _prime_pick_place_transport(env)

    env.set_manual_configuration(
        gripper_xyz=(PICK_PLACE_GOAL_POS[0], PICK_PLACE_GOAL_POS[1], 0.16),
        finger_positions=(0.0, 0.0),
        object_position=PICK_PLACE_GOAL_POS,
    )

    success = False
    for step_index in range(env.success_hold_steps):
        _, _, terminated, truncated, info = env.step(_pick_place_noop(env))
        assert not truncated
        if step_index < env.success_hold_steps - 1:
            assert not terminated
        if terminated:
            success = bool(info["is_success"])
            break

    assert success
    env.close()


def test_arm_reward_uses_contact_and_alignment_shaping() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="grasp_lift",
            observation_mode="contact",
            max_episode_steps=50,
        ),
        RewardConfig(contact_weight=0.25, grasp_alignment_weight=1.0),
    )

    no_contact_arm, no_contact_fingers, _ = ARM_CONTACT_POSES["none"]
    both_contact_arm, both_contact_fingers, _ = ARM_CONTACT_POSES["both"]

    env.set_manual_configuration(
        arm_joint_positions=no_contact_arm,
        finger_positions=no_contact_fingers,
        object_position=LIFT_OBJECT_POS,
    )
    no_contact = env._compute_potential_terms(env._get_true_contact_bits())

    env.set_manual_configuration(
        arm_joint_positions=both_contact_arm,
        finger_positions=both_contact_fingers,
        object_position=LIFT_OBJECT_POS,
    )
    with_contact = env._compute_potential_terms(env._get_true_contact_bits())

    env.set_manual_configuration(
        arm_joint_positions=both_contact_arm,
        finger_positions=both_contact_fingers,
        object_position=(0.0, 0.0, 0.16),
    )
    lifted = env._compute_potential_terms(env._get_true_contact_bits())

    assert with_contact.contact > no_contact.contact
    assert with_contact.grasp_alignment > no_contact.grasp_alignment
    assert lifted.lift > with_contact.lift
    env.close()


@pytest.mark.parametrize(
    ("embodiment", "task", "lifted_state"),
    [
        (
            "cartesian_gripper",
            "grasp_lift",
            {
                "gripper_xyz": (0.0, 0.0, 0.15),
                "finger_positions": (0.048, 0.048),
                "object_position": (0.0, 0.0, 0.16),
            },
        ),
        (
            "arm_pinch",
            "grasp_lift",
            {
                "arm_joint_positions": ARM_CONTACT_POSES["both"][0],
                "finger_positions": ARM_CONTACT_POSES["both"][1],
                "object_position": (0.0, 0.0, 0.16),
            },
        ),
    ],
)
def test_shaped_reward_does_not_accumulate_on_plateau(
    embodiment: str,
    task: str,
    lifted_state: dict[str, object],
) -> None:
    env = create_env(embodiment=embodiment, task=task)
    env.set_manual_configuration(**lifted_state)

    potential = env._compute_potential_terms(env._get_true_contact_bits())
    reward_terms = env._compute_reward_terms(
        np.zeros(env.action_space.shape[0], dtype=np.float32),
        potential,
        potential,
        success=False,
    )

    assert reward_terms.total == 0.0
    env.close()


def test_scripted_cartesian_grasp_sequence_reaches_success() -> None:
    env = ContactAwareGraspLiftEnv(
        EnvConfig(task="grasp_lift", observation_mode="contact", max_episode_steps=200),
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


def test_scripted_arm_lift_state_reaches_success() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="grasp_lift",
            observation_mode="contact",
            max_episode_steps=200,
        ),
        RewardConfig(),
    )
    env.reset(seed=0)
    lifted_object_pos = np.array([0.0, 0.0, 0.16], dtype=np.float64)
    env.set_manual_configuration(
        arm_joint_positions=np.array([-0.0241, 0.9398, -1.1581, -1.7], dtype=np.float64),
        finger_positions=ARM_CONTACT_POSES["both"][1],
        object_position=lifted_object_pos,
    )
    status = env._build_task_status(env._get_true_contact_bits())
    assert status.is_grasped
    assert lifted_object_pos[2] >= env._success_height

    for _ in range(env.success_hold_steps):
        env._update_success_tracking(object_pos=lifted_object_pos, task_status=status)

    assert env._success_streak >= env.success_hold_steps
    env.close()


def test_scripted_arm_pick_place_sequence_reaches_success() -> None:
    env = ArmPinchGraspLiftEnv(
        EnvConfig(
            embodiment="arm_pinch",
            task="pick_place_ab",
            observation_mode="contact",
            max_episode_steps=250,
        ),
        RewardConfig(),
    )
    env.reset(seed=0)
    env.set_manual_configuration(
        arm_joint_positions=ARM_PICK_PLACE_START_CONTACT_POSE,
        finger_positions=ARM_PICK_PLACE_START_CONTACT_FINGERS,
        object_position=PICK_PLACE_START_POS,
    )
    status = env._build_task_status(env._get_true_contact_bits())
    assert status.is_grasped
    env._update_success_tracking(object_pos=env._get_object_state()[0], task_status=status)

    lift_height = PICK_PLACE_START_POS[2] + 0.06
    env.set_manual_configuration(
        arm_joint_positions=np.asarray(env.env_config.initial_arm_joint_positions, dtype=np.float64),
        finger_positions=(0.0, 0.0),
        object_position=(PICK_PLACE_START_POS[0], PICK_PLACE_START_POS[1], lift_height),
    )
    status = TaskStatus(
        has_any_contact=True,
        has_dual_contact=True,
        lift_clearance=lift_height - PICK_PLACE_START_POS[2],
        goal_distance_xy=float(
            np.linalg.norm(np.asarray(PICK_PLACE_START_POS[:2]) - np.asarray(PICK_PLACE_GOAL_POS[:2]))
        ),
        goal_height_error=lift_height - PICK_PLACE_START_POS[2],
        grasp_alignment_score=1.0,
        is_grasped=True,
        is_lifted_grasp=True,
    )
    env._update_success_tracking(object_pos=env._get_object_state()[0], task_status=status)
    assert env._episode_has_lifted_for_transport

    env.set_manual_configuration(
        arm_joint_positions=np.asarray(env.env_config.initial_arm_joint_positions, dtype=np.float64),
        finger_positions=(0.0, 0.0),
        object_position=PICK_PLACE_GOAL_POS,
    )
    for _ in range(env.success_hold_steps):
        status = env._build_task_status(env._get_true_contact_bits())
        env._update_success_tracking(object_pos=env._get_object_state()[0], task_status=status)

    assert env._episode_has_placed
    assert env._episode_has_released
    assert env._episode_has_settled
    assert env._success_streak >= env.success_hold_steps
    env.close()
