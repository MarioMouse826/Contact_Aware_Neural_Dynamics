from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .config import EnvConfig, RewardConfig

SUPPORTED_EMBODIMENTS = {"cartesian_gripper", "arm_pinch"}
SUPPORTED_OBSERVATION_MODES = {"baseline", "contact"}
SUPPORTED_CONTACT_OVERRIDES = {None, "ones", "zeros"}


@dataclass(frozen=True)
class PotentialTerms:
    reach: float
    contact: float
    lift: float
    hold: float

    @property
    def total(self) -> float:
        return self.reach + self.contact + self.lift + self.hold


@dataclass(frozen=True)
class RewardTerms:
    reach: float
    contact: float
    lift: float
    hold: float
    success_bonus: float
    action_penalty: float
    current_potential: PotentialTerms
    previous_potential: PotentialTerms

    @property
    def total(self) -> float:
        return (
            self.reach
            + self.contact
            + self.lift
            + self.hold
            + self.success_bonus
            - self.action_penalty
        )


def validate_env_config(env_config: EnvConfig) -> None:
    if env_config.embodiment not in SUPPORTED_EMBODIMENTS:
        raise ValueError(
            f"Unsupported embodiment '{env_config.embodiment}'. "
            f"Expected one of {sorted(SUPPORTED_EMBODIMENTS)}."
        )
    if env_config.observation_mode not in SUPPORTED_OBSERVATION_MODES:
        raise ValueError(
            f"Unsupported observation mode '{env_config.observation_mode}'. "
            f"Expected one of {sorted(SUPPORTED_OBSERVATION_MODES)}."
        )
    if env_config.contact_override not in SUPPORTED_CONTACT_OVERRIDES:
        raise ValueError(
            f"Unsupported contact override '{env_config.contact_override}'. "
            f"Expected one of {sorted(value for value in SUPPORTED_CONTACT_OVERRIDES if value is not None)} "
            "or None."
        )
    if env_config.embodiment == "arm_pinch" and env_config.contact_override == "ones":
        raise ValueError(
            "The articulated arm task does not support always-contact training."
        )
    if len(env_config.arm_joint_delta_scales) != 4:
        raise ValueError(
            "EnvConfig.arm_joint_delta_scales must contain exactly four values."
        )
    if len(env_config.initial_arm_joint_positions) != 4:
        raise ValueError(
            "EnvConfig.initial_arm_joint_positions must contain exactly four values."
        )


class BaseContactAwareEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.env_config = env_config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()
        validate_env_config(self.env_config)

        xml = self._build_mjcf()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self._controlled_joint_names = list(self._controlled_joint_names_impl())
        self._controlled_joint_ids = [
            self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self._controlled_joint_names
        ]
        self._controlled_qpos_adr = np.asarray(
            [self.model.jnt_qposadr[joint_id] for joint_id in self._controlled_joint_ids],
            dtype=np.int32,
        )
        self._controlled_qvel_adr = np.asarray(
            [self.model.jnt_dofadr[joint_id] for joint_id in self._controlled_joint_ids],
            dtype=np.int32,
        )

        if self.model.nu != len(self._controlled_joint_names):
            raise ValueError(
                f"Expected {len(self._controlled_joint_names)} actuators, found {self.model.nu}."
            )

        self._object_joint_id = self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self._object_qpos_adr = int(self.model.jnt_qposadr[self._object_joint_id])
        self._object_qvel_adr = int(self.model.jnt_dofadr[self._object_joint_id])

        self._left_pad_geom_id = self._name_to_id(
            mujoco.mjtObj.mjOBJ_GEOM, "left_pad_geom"
        )
        self._right_pad_geom_id = self._name_to_id(
            mujoco.mjtObj.mjOBJ_GEOM, "right_pad_geom"
        )
        self._object_geom_id = self._name_to_id(
            mujoco.mjtObj.mjOBJ_GEOM, "object_geom"
        )
        self._left_tip_site_id = self._name_to_id(
            mujoco.mjtObj.mjOBJ_SITE, "left_tip_site"
        )
        self._right_tip_site_id = self._name_to_id(
            mujoco.mjtObj.mjOBJ_SITE, "right_tip_site"
        )

        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._target_ctrl = np.zeros(self.model.nu, dtype=np.float64)

        self._lift_start_height = (
            self.env_config.table_height + self.env_config.object_half_extents[2] + 0.01
        )
        self._success_height = (
            self.env_config.table_height + self.env_config.success_height_over_table
        )
        self._lift_target_span = max(1e-6, self._success_height - self._lift_start_height)

        control_dim = len(self._controlled_joint_names)
        obs_dim = (2 * control_dim) + 16
        if self.env_config.observation_mode == "contact":
            obs_dim += 2

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._action_size(),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._episode_steps = 0
        self._episode_both_contact_steps = 0
        self._episode_max_height = 0.0
        self._success_streak = 0
        self._episode_steps_above_success_height = 0
        self._episode_best_success_streak = 0
        self._previous_potential = PotentialTerms(0.0, 0.0, 0.0, 0.0)

    def _controlled_joint_names_impl(self) -> list[str]:
        raise NotImplementedError

    def _action_size(self) -> int:
        raise NotImplementedError

    def _build_mjcf(self) -> str:
        raise NotImplementedError

    def _set_targets_from_action(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def _reset_manipulator(
        self,
        *,
        object_position: np.ndarray,
        object_quat: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def _name_to_id(self, obj_type: int, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, obj_type, name)
        if identifier < 0:
            raise ValueError(f"Failed to resolve MuJoCo object '{name}'.")
        return int(identifier)

    def _coerce_finger_positions(
        self, finger_positions: tuple[float, float] | float
    ) -> tuple[float, float]:
        if isinstance(finger_positions, (float, int)):
            return float(finger_positions), float(finger_positions)
        return float(finger_positions[0]), float(finger_positions[1])

    def _coerce_object_quat(
        self,
        object_quat: np.ndarray | tuple[float, float, float, float] | None,
    ) -> np.ndarray:
        if object_quat is None:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.asarray(object_quat, dtype=np.float64)

    def _apply_manual_configuration(
        self,
        *,
        controlled_state: np.ndarray,
        object_position: tuple[float, float, float] | np.ndarray,
        object_quat: np.ndarray | tuple[float, float, float, float] | None = None,
        zero_vel: bool = True,
    ) -> None:
        controlled_state = np.asarray(controlled_state, dtype=np.float64)
        if controlled_state.shape != (len(self._controlled_joint_names),):
            raise ValueError(
                f"Expected controlled state shape {(len(self._controlled_joint_names),)}, "
                f"got {controlled_state.shape}."
            )

        self.data.qpos[self._controlled_qpos_adr] = controlled_state
        self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 3] = np.asarray(
            object_position,
            dtype=np.float64,
        )
        self.data.qpos[self._object_qpos_adr + 3 : self._object_qpos_adr + 7] = (
            self._coerce_object_quat(object_quat)
        )
        if zero_vel:
            self.data.qvel[:] = 0.0

        self._target_ctrl = np.clip(controlled_state, self._ctrl_low, self._ctrl_high)
        self.data.ctrl[:] = self._target_ctrl
        mujoco.mj_forward(self.model, self.data)

    def _get_control_state(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.data.qpos[self._controlled_qpos_adr].copy()
        qdot = self.data.qvel[self._controlled_qvel_adr].copy()
        return q, qdot

    def _get_object_state(self) -> tuple[np.ndarray, np.ndarray]:
        qpos = self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 7]
        return qpos[:3].copy(), qpos[3:].copy()

    def _get_true_contact_bits(self) -> np.ndarray:
        bits = np.zeros(2, dtype=np.float32)
        for index in range(self.data.ncon):
            geom_a, geom_b = self.data.contact[index].geom
            geom_pair = {int(geom_a), int(geom_b)}
            if self._object_geom_id not in geom_pair:
                continue
            if self._left_pad_geom_id in geom_pair:
                bits[0] = 1.0
            if self._right_pad_geom_id in geom_pair:
                bits[1] = 1.0
        return bits

    def _get_observed_contact_bits(self, true_bits: np.ndarray) -> np.ndarray:
        if self.env_config.contact_override == "ones":
            return np.ones_like(true_bits, dtype=np.float32)
        if self.env_config.contact_override == "zeros":
            return np.zeros_like(true_bits, dtype=np.float32)
        return true_bits.astype(np.float32, copy=True)

    def _get_fingertip_positions(self) -> tuple[np.ndarray, np.ndarray]:
        left = self.data.site_xpos[self._left_tip_site_id].copy()
        right = self.data.site_xpos[self._right_tip_site_id].copy()
        return left, right

    def _get_observation(self, true_contact_bits: np.ndarray | None = None) -> np.ndarray:
        true_contact_bits = (
            self._get_true_contact_bits() if true_contact_bits is None else true_contact_bits
        )
        observed_contact_bits = self._get_observed_contact_bits(true_contact_bits)
        q, qdot = self._get_control_state()
        object_pos, object_quat = self._get_object_state()
        left_tip, right_tip = self._get_fingertip_positions()
        gripper_center = 0.5 * (left_tip + right_tip)
        delta = object_pos - gripper_center

        obs_parts = [
            q.astype(np.float32),
            qdot.astype(np.float32),
            object_pos.astype(np.float32),
            object_quat.astype(np.float32),
            left_tip.astype(np.float32),
            right_tip.astype(np.float32),
            delta.astype(np.float32),
        ]
        if self.env_config.observation_mode == "contact":
            obs_parts.append(observed_contact_bits.astype(np.float32))
        return np.concatenate(obs_parts).astype(np.float32)

    def _compute_contact_potential(self, true_contact_bits: np.ndarray) -> float:
        return self.reward_config.contact_weight * float(np.all(true_contact_bits > 0.5))

    def _compute_potential_terms(self, true_contact_bits: np.ndarray) -> PotentialTerms:
        object_pos, _ = self._get_object_state()
        left_tip, right_tip = self._get_fingertip_positions()
        gripper_center = 0.5 * (left_tip + right_tip)
        reach_distance = float(np.linalg.norm(gripper_center - object_pos))
        reach = self.reward_config.reach_weight * math.exp(-10.0 * reach_distance)
        contact = self._compute_contact_potential(true_contact_bits)
        lift_progress = float(
            np.clip(
                (object_pos[2] - self._lift_start_height) / self._lift_target_span,
                0.0,
                1.0,
            )
        )
        lift = self.reward_config.lift_weight * lift_progress
        hold_progress = float(
            np.clip(
                self._success_streak / max(1, self.env_config.success_hold_steps),
                0.0,
                1.0,
            )
        )
        hold = self.reward_config.hold_weight * hold_progress
        return PotentialTerms(reach=reach, contact=contact, lift=lift, hold=hold)

    def _compute_reward_terms(
        self,
        action: np.ndarray,
        previous_potential: PotentialTerms,
        current_potential: PotentialTerms,
        *,
        success: bool,
    ) -> RewardTerms:
        action_penalty = self.reward_config.action_penalty_weight * float(
            np.square(action).sum()
        )
        success_bonus = self.reward_config.success_bonus if success else 0.0
        return RewardTerms(
            reach=current_potential.reach - previous_potential.reach,
            contact=current_potential.contact - previous_potential.contact,
            lift=current_potential.lift - previous_potential.lift,
            hold=current_potential.hold - previous_potential.hold,
            success_bonus=success_bonus,
            action_penalty=action_penalty,
            current_potential=current_potential,
            previous_potential=previous_potential,
        )

    def _build_info(
        self,
        *,
        true_contact_bits: np.ndarray,
        observed_contact_bits: np.ndarray,
        reward_terms: RewardTerms,
        success: bool,
        termination_reason: str | None = None,
    ) -> dict[str, Any]:
        object_pos, _ = self._get_object_state()
        info: dict[str, Any] = {
            "is_success": float(success),
            "contact_stability": float(
                self._episode_both_contact_steps / max(1, self._episode_steps)
            ),
            "max_lift_height": float(self._episode_max_height),
            "object_height": float(object_pos[2] - self.env_config.table_height),
            "best_success_streak": int(self._episode_best_success_streak),
            "steps_above_success_height": int(self._episode_steps_above_success_height),
            "threshold_crossed": float(self._episode_steps_above_success_height > 0),
            "true_contact_left": float(true_contact_bits[0]),
            "true_contact_right": float(true_contact_bits[1]),
            "observed_contact_left": float(observed_contact_bits[0]),
            "observed_contact_right": float(observed_contact_bits[1]),
            "reward/reach_delta": reward_terms.reach,
            "reward/contact_delta": reward_terms.contact,
            "reward/lift_delta": reward_terms.lift,
            "reward/hold_delta": reward_terms.hold,
            "reward/success_bonus": reward_terms.success_bonus,
            "reward/action_penalty": reward_terms.action_penalty,
            "reward/total": reward_terms.total,
            "potential/reach": reward_terms.current_potential.reach,
            "potential/contact": reward_terms.current_potential.contact,
            "potential/lift": reward_terms.current_potential.lift,
            "potential/hold": reward_terms.current_potential.hold,
        }
        if termination_reason is not None:
            info["termination_reason"] = termination_reason
        return info

    def _object_dropped(self, object_pos: np.ndarray) -> bool:
        too_low = object_pos[2] < (
            self.env_config.table_height - self.env_config.termination_drop_margin
        )
        too_far = np.linalg.norm(object_pos[:2]) > 0.35
        return bool(too_low or too_far)

    def _sample_object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        object_xy_noise = self.env_config.reset_object_xy_range
        object_x = float(self.np_random.uniform(-object_xy_noise, object_xy_noise))
        object_y = float(self.np_random.uniform(-object_xy_noise, object_xy_noise))
        object_yaw = float(
            self.np_random.uniform(
                -self.env_config.reset_object_yaw_range,
                self.env_config.reset_object_yaw_range,
            )
        )
        object_quat = np.array(
            [math.cos(object_yaw / 2.0), 0.0, 0.0, math.sin(object_yaw / 2.0)],
            dtype=np.float64,
        )
        object_z = self.env_config.table_height + self.env_config.object_half_extents[2]
        object_position = np.array([object_x, object_y, object_z], dtype=np.float64)
        return object_position, object_quat

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        object_position, object_quat = self._sample_object_pose()
        self._reset_manipulator(object_position=object_position, object_quat=object_quat)

        self._episode_steps = 0
        self._episode_both_contact_steps = 0
        self._episode_max_height = float(object_position[2] - self.env_config.table_height)
        self._success_streak = 0
        self._episode_steps_above_success_height = 0
        self._episode_best_success_streak = 0

        true_contact_bits = self._get_true_contact_bits()
        observed_contact_bits = self._get_observed_contact_bits(true_contact_bits)
        self._previous_potential = self._compute_potential_terms(true_contact_bits)
        reward_terms = RewardTerms(
            reach=0.0,
            contact=0.0,
            lift=0.0,
            hold=0.0,
            success_bonus=0.0,
            action_penalty=0.0,
            current_potential=self._previous_potential,
            previous_potential=self._previous_potential,
        )
        observation = self._get_observation(true_contact_bits)
        info = self._build_info(
            true_contact_bits=true_contact_bits,
            observed_contact_bits=observed_contact_bits,
            reward_terms=reward_terms,
            success=False,
        )
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_targets_from_action(action)

        for _ in range(self.env_config.substeps):
            mujoco.mj_step(self.model, self.data)

        self._episode_steps += 1

        true_contact_bits = self._get_true_contact_bits()
        observed_contact_bits = self._get_observed_contact_bits(true_contact_bits)
        object_pos, _ = self._get_object_state()
        object_height = float(object_pos[2] - self.env_config.table_height)
        self._episode_max_height = max(self._episode_max_height, object_height)

        if bool(np.all(true_contact_bits > 0.5)):
            self._episode_both_contact_steps += 1

        if object_pos[2] >= self._success_height:
            self._success_streak += 1
            self._episode_steps_above_success_height += 1
        else:
            self._success_streak = 0
        self._episode_best_success_streak = max(
            self._episode_best_success_streak,
            self._success_streak,
        )

        success = self._success_streak >= self.env_config.success_hold_steps
        current_potential = self._compute_potential_terms(true_contact_bits)
        reward_terms = self._compute_reward_terms(
            action,
            self._previous_potential,
            current_potential,
            success=success,
        )
        self._previous_potential = current_potential

        terminated = False
        termination_reason: str | None = None
        if success:
            terminated = True
            termination_reason = "success"
        elif self._object_dropped(object_pos):
            terminated = True
            termination_reason = "dropped"

        truncated = self._episode_steps >= self.env_config.max_episode_steps
        if truncated and termination_reason is None:
            termination_reason = "time_limit"

        observation = self._get_observation(true_contact_bits)
        info = self._build_info(
            true_contact_bits=true_contact_bits,
            observed_contact_bits=observed_contact_bits,
            reward_terms=reward_terms,
            success=success,
            termination_reason=termination_reason,
        )

        return observation, float(reward_terms.total), terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


class ContactAwareGraspLiftEnv(BaseContactAwareEnv):
    def __init__(
        self,
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self._workspace_xy_limit = 0.18
        self._workspace_z = (0.08, 0.25)
        super().__init__(env_config=env_config, reward_config=reward_config)

    def _controlled_joint_names_impl(self) -> list[str]:
        return [
            "grip_x",
            "grip_y",
            "grip_z",
            "left_finger_slide",
            "right_finger_slide",
        ]

    def _action_size(self) -> int:
        return 4

    def _build_mjcf(self) -> str:
        object_half_extents = " ".join(
            f"{value:.5f}" for value in self.env_config.object_half_extents
        )
        object_friction = " ".join(
            f"{value:.5f}" for value in self.env_config.object_friction
        )
        finger_friction = " ".join(
            f"{value:.5f}" for value in self.env_config.finger_friction
        )
        table_half_height = self.env_config.table_height / 2.0
        object_z = self.env_config.table_height + self.env_config.object_half_extents[2]

        return f"""
<mujoco model="contact_aware_grasp_lift">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4" cone="elliptic"/>
  <size nstack="300000"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <default>
    <joint damping="4" armature="0.01"/>
    <geom solref="0.003 1" solimp="0.95 0.99 0.001" condim="4"/>
    <position kp="400" ctrllimited="true"/>
  </default>
  <worldbody>
    <light diffuse="0.8 0.8 0.8" pos="0 0 1.5"/>
    <camera name="overview" pos="0 -0.55 0.38" xyaxes="1 0 0 0 0.6 0.8"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.93 0.93 0.93 1"/>
    <body name="table" pos="0 0 {table_half_height:.5f}">
      <geom name="table_geom" type="box" size="0.30 0.30 {table_half_height:.5f}" rgba="0.56 0.47 0.35 1"/>
    </body>
    <body name="object" pos="0 0 {object_z:.5f}">
      <freejoint name="object_free"/>
      <geom
        name="object_geom"
        type="box"
        size="{object_half_extents}"
        mass="{self.env_config.object_mass:.5f}"
        friction="{object_friction}"
        rgba="0.2 0.45 0.85 1"
      />
    </body>
    <body name="gripper" pos="0 0 0">
      <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
      <joint name="grip_x" type="slide" axis="1 0 0" range="-0.16 0.16"/>
      <joint name="grip_y" type="slide" axis="0 1 0" range="-0.16 0.16"/>
      <joint name="grip_z" type="slide" axis="0 0 1" range="{self._workspace_z[0]:.5f} {self._workspace_z[1]:.5f}"/>
      <body name="left_finger" pos="-0.08 0 0">
        <joint name="left_finger_slide" type="slide" axis="1 0 0" range="0 0.06"/>
        <geom name="left_pad_geom" type="box" size="0.01 0.03 0.04" friction="{finger_friction}" rgba="0.95 0.4 0.35 1"/>
        <site name="left_tip_site" pos="0.01 0 0" size="0.004" rgba="0.1 0.8 0.1 1"/>
      </body>
      <body name="right_finger" pos="0.08 0 0">
        <joint name="right_finger_slide" type="slide" axis="-1 0 0" range="0 0.06"/>
        <geom name="right_pad_geom" type="box" size="0.01 0.03 0.04" friction="{finger_friction}" rgba="0.35 0.4 0.95 1"/>
        <site name="right_tip_site" pos="-0.01 0 0" size="0.004" rgba="0.1 0.8 0.1 1"/>
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="left_finger" body2="right_finger"/>
  </contact>
  <actuator>
    <position name="x_act" joint="grip_x" ctrlrange="-0.16 0.16"/>
    <position name="y_act" joint="grip_y" ctrlrange="-0.16 0.16"/>
    <position name="z_act" joint="grip_z" ctrlrange="{self._workspace_z[0]:.5f} {self._workspace_z[1]:.5f}"/>
    <position name="left_act" joint="left_finger_slide" ctrlrange="0 0.06" kp="300"/>
    <position name="right_act" joint="right_finger_slide" ctrlrange="0 0.06" kp="300"/>
  </actuator>
</mujoco>
"""

    def _set_targets_from_action(self, action: np.ndarray) -> None:
        deltas = np.array(
            [
                action[0] * self.env_config.action_scale_xyz,
                action[1] * self.env_config.action_scale_xyz,
                action[2] * self.env_config.action_scale_xyz,
                action[3] * self.env_config.action_scale_grip,
                action[3] * self.env_config.action_scale_grip,
            ],
            dtype=np.float64,
        )
        self._target_ctrl = np.clip(self._target_ctrl + deltas, self._ctrl_low, self._ctrl_high)
        self.data.ctrl[:] = self._target_ctrl

    def _reset_manipulator(
        self,
        *,
        object_position: np.ndarray,
        object_quat: np.ndarray,
    ) -> None:
        gripper_x = float(
            np.clip(
                object_position[0]
                + self.np_random.uniform(
                    -self.env_config.reset_gripper_xy_noise,
                    self.env_config.reset_gripper_xy_noise,
                ),
                -self._workspace_xy_limit,
                self._workspace_xy_limit,
            )
        )
        gripper_y = float(
            np.clip(
                object_position[1]
                + self.np_random.uniform(
                    -self.env_config.reset_gripper_xy_noise,
                    self.env_config.reset_gripper_xy_noise,
                ),
                -self._workspace_xy_limit,
                self._workspace_xy_limit,
            )
        )
        gripper_z = float(
            np.clip(
                self.env_config.initial_gripper_height
                + self.np_random.uniform(
                    -self.env_config.reset_gripper_z_noise,
                    self.env_config.reset_gripper_z_noise,
                ),
                self._workspace_z[0],
                self._workspace_z[1],
            )
        )
        self.set_manual_configuration(
            gripper_xyz=(gripper_x, gripper_y, gripper_z),
            finger_positions=(
                self.env_config.initial_finger_position,
                self.env_config.initial_finger_position,
            ),
            object_position=object_position,
            object_quat=object_quat,
            zero_vel=True,
        )

    def set_manual_configuration(
        self,
        *,
        gripper_xyz: tuple[float, float, float],
        finger_positions: tuple[float, float] | float,
        object_position: tuple[float, float, float] | np.ndarray,
        object_quat: np.ndarray | tuple[float, float, float, float] | None = None,
        zero_vel: bool = True,
    ) -> None:
        left_finger, right_finger = self._coerce_finger_positions(finger_positions)
        controlled_state = np.array(
            [
                float(gripper_xyz[0]),
                float(gripper_xyz[1]),
                float(gripper_xyz[2]),
                left_finger,
                right_finger,
            ],
            dtype=np.float64,
        )
        self._apply_manual_configuration(
            controlled_state=controlled_state,
            object_position=object_position,
            object_quat=object_quat,
            zero_vel=zero_vel,
        )


class ArmPinchGraspLiftEnv(BaseContactAwareEnv):
    def __init__(
        self,
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self._arm_base_position = np.array([0.0, -0.18, 0.18], dtype=np.float64)
        super().__init__(env_config=env_config, reward_config=reward_config)

    def _controlled_joint_names_impl(self) -> list[str]:
        return [
            "base_yaw",
            "shoulder_pitch",
            "elbow_pitch",
            "wrist_pitch",
            "left_finger_slide",
            "right_finger_slide",
        ]

    def _action_size(self) -> int:
        return 5

    def _build_mjcf(self) -> str:
        object_half_extents = " ".join(
            f"{value:.5f}" for value in self.env_config.object_half_extents
        )
        object_friction = " ".join(
            f"{value:.5f}" for value in self.env_config.object_friction
        )
        finger_friction = " ".join(
            f"{value:.5f}" for value in self.env_config.finger_friction
        )
        table_half_height = self.env_config.table_height / 2.0
        object_z = self.env_config.table_height + self.env_config.object_half_extents[2]

        return f"""
<mujoco model="contact_aware_arm_pinch">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4" cone="elliptic"/>
  <size nstack="300000"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <default>
    <joint damping="3" armature="0.01"/>
    <geom solref="0.003 1" solimp="0.95 0.99 0.001" condim="4"/>
    <position kp="120" ctrllimited="true"/>
  </default>
  <worldbody>
    <light diffuse="0.8 0.8 0.8" pos="0 0 1.5"/>
    <camera name="overview" pos="0 -0.55 0.38" xyaxes="1 0 0 0 0.6 0.8"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.93 0.93 0.93 1"/>
    <body name="table" pos="0 0 {table_half_height:.5f}">
      <geom name="table_geom" type="box" size="0.30 0.30 {table_half_height:.5f}" rgba="0.56 0.47 0.35 1"/>
    </body>
    <body name="object" pos="0 0 {object_z:.5f}">
      <freejoint name="object_free"/>
      <geom
        name="object_geom"
        type="box"
        size="{object_half_extents}"
        mass="{self.env_config.object_mass:.5f}"
        friction="{object_friction}"
        rgba="0.2 0.45 0.85 1"
      />
    </body>
    <body name="arm_base" pos="{self._arm_base_position[0]:.5f} {self._arm_base_position[1]:.5f} {self._arm_base_position[2]:.5f}">
      <inertial pos="0 0 0" mass="1.2" diaginertia="0.005 0.005 0.005"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" range="-1.20 1.20"/>
      <body name="shoulder_link" pos="0 0 0">
        <joint name="shoulder_pitch" type="hinge" axis="1 0 0" range="-1.60 1.00"/>
        <geom type="capsule" fromto="0 0 0 0 0.14 0" size="0.022" rgba="0.25 0.25 0.25 1"/>
        <body name="elbow_link" pos="0 0.14 0">
          <joint name="elbow_pitch" type="hinge" axis="1 0 0" range="-1.70 1.70"/>
          <geom type="capsule" fromto="0 0 0 0 0.14 0" size="0.018" rgba="0.35 0.35 0.35 1"/>
          <body name="wrist_link" pos="0 0.14 0">
            <joint name="wrist_pitch" type="hinge" axis="1 0 0" range="-1.70 1.70"/>
            <geom type="capsule" fromto="0 0 0 0 0.08 0" size="0.014" rgba="0.45 0.45 0.45 1"/>
            <body name="hand" pos="0 0.08 0">
              <geom name="palm_geom" type="box" pos="0 0.015 0" size="0.03 0.02 0.012" rgba="0.70 0.70 0.70 1"/>
              <body name="left_finger" pos="-0.05 0.03 0">
                <joint name="left_finger_slide" type="slide" axis="1 0 0" range="0 0.03"/>
                <geom name="left_pad_geom" type="box" pos="0.01 0 0" size="0.008 0.02 0.035" friction="{finger_friction}" rgba="0.95 0.4 0.35 1"/>
                <site name="left_tip_site" pos="0.018 0 0" size="0.004" rgba="0.1 0.8 0.1 1"/>
              </body>
              <body name="right_finger" pos="0.05 0.03 0">
                <joint name="right_finger_slide" type="slide" axis="-1 0 0" range="0 0.03"/>
                <geom name="right_pad_geom" type="box" pos="-0.01 0 0" size="0.008 0.02 0.035" friction="{finger_friction}" rgba="0.35 0.4 0.95 1"/>
                <site name="right_tip_site" pos="-0.018 0 0" size="0.004" rgba="0.1 0.8 0.1 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="left_finger" body2="right_finger"/>
  </contact>
  <actuator>
    <position name="base_yaw_act" joint="base_yaw" ctrlrange="-1.20 1.20" kp="100"/>
    <position name="shoulder_pitch_act" joint="shoulder_pitch" ctrlrange="-1.60 1.00" kp="110"/>
    <position name="elbow_pitch_act" joint="elbow_pitch" ctrlrange="-1.70 1.70" kp="110"/>
    <position name="wrist_pitch_act" joint="wrist_pitch" ctrlrange="-1.70 1.70" kp="90"/>
    <position name="left_finger_act" joint="left_finger_slide" ctrlrange="0 0.03" kp="250"/>
    <position name="right_finger_act" joint="right_finger_slide" ctrlrange="0 0.03" kp="250"/>
  </actuator>
</mujoco>
"""

    def _set_targets_from_action(self, action: np.ndarray) -> None:
        joint_scales = np.asarray(
            self.env_config.arm_joint_delta_scales,
            dtype=np.float64,
        )
        deltas = np.array(
            [
                action[0] * joint_scales[0],
                action[1] * joint_scales[1],
                action[2] * joint_scales[2],
                action[3] * joint_scales[3],
                action[4] * self.env_config.action_scale_grip,
                action[4] * self.env_config.action_scale_grip,
            ],
            dtype=np.float64,
        )
        self._target_ctrl = np.clip(self._target_ctrl + deltas, self._ctrl_low, self._ctrl_high)
        self.data.ctrl[:] = self._target_ctrl

    def _compute_contact_potential(self, true_contact_bits: np.ndarray) -> float:
        return 0.0

    def _reset_manipulator(
        self,
        *,
        object_position: np.ndarray,
        object_quat: np.ndarray,
    ) -> None:
        arm_joint_positions = np.asarray(
            self.env_config.initial_arm_joint_positions,
            dtype=np.float64,
        )
        arm_joint_positions = arm_joint_positions + self.np_random.uniform(
            -self.env_config.reset_arm_joint_noise,
            self.env_config.reset_arm_joint_noise,
            size=arm_joint_positions.shape,
        )
        arm_joint_positions = np.clip(
            arm_joint_positions,
            self._ctrl_low[:4],
            self._ctrl_high[:4],
        )
        initial_finger_position = float(
            np.clip(
                self.env_config.initial_finger_position,
                self._ctrl_low[4],
                self._ctrl_high[4],
            )
        )
        self.set_manual_configuration(
            arm_joint_positions=arm_joint_positions,
            finger_positions=(initial_finger_position, initial_finger_position),
            object_position=object_position,
            object_quat=object_quat,
            zero_vel=True,
        )

    def set_manual_configuration(
        self,
        *,
        arm_joint_positions: tuple[float, float, float, float] | np.ndarray,
        finger_positions: tuple[float, float] | float,
        object_position: tuple[float, float, float] | np.ndarray,
        object_quat: np.ndarray | tuple[float, float, float, float] | None = None,
        zero_vel: bool = True,
    ) -> None:
        arm_joint_positions = np.asarray(arm_joint_positions, dtype=np.float64)
        if arm_joint_positions.shape != (4,):
            raise ValueError(
                f"Expected four arm joint positions, got shape {arm_joint_positions.shape}."
            )
        left_finger, right_finger = self._coerce_finger_positions(finger_positions)
        controlled_state = np.array(
            [
                float(arm_joint_positions[0]),
                float(arm_joint_positions[1]),
                float(arm_joint_positions[2]),
                float(arm_joint_positions[3]),
                left_finger,
                right_finger,
            ],
            dtype=np.float64,
        )
        self._apply_manual_configuration(
            controlled_state=controlled_state,
            object_position=object_position,
            object_quat=object_quat,
            zero_vel=zero_vel,
        )


def make_env(
    env_config: EnvConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> BaseContactAwareEnv:
    env_config = env_config or EnvConfig()
    reward_config = reward_config or RewardConfig()
    validate_env_config(env_config)

    if env_config.embodiment == "cartesian_gripper":
        return ContactAwareGraspLiftEnv(env_config=env_config, reward_config=reward_config)
    if env_config.embodiment == "arm_pinch":
        return ArmPinchGraspLiftEnv(env_config=env_config, reward_config=reward_config)
    raise ValueError(f"Unsupported embodiment '{env_config.embodiment}'.")
