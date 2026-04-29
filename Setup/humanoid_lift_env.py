"""
Humanoid Lift Environment — Unitree G1 walking to a block and lifting it.

Task: G1 spawned 5m from a red block. Walk to it, grasp, lift >10cm for 1s.

Observation augmentation: 4 binary contact bits indicating whether each of
{left_palm, left_fingers, right_palm, right_fingers} is touching the block.

Set ENV_USE_CONTACT_BITS=0 to ablate contact bits (baseline).
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MENAGERIE_PATH = REPO_ROOT / "mujoco_menagerie"
MENAGERIE_PATH = Path(os.environ.get("MENAGERIE_PATH", DEFAULT_MENAGERIE_PATH))
G1_SCENE_RELPATH = "unitree_g1/scene_with_hands.xml"

MAX_EPISODE_STEPS = 1000
CONTROL_FREQ_HZ = 50
SIM_STEPS_PER_CONTROL = 5

BLOCK_INITIAL_POS = np.array([0.5, 0.0, 0.025])
BLOCK_SIZE = 0.05
SUCCESS_LIFT_HEIGHT = 0.10
SUCCESS_HOLD_STEPS = int(1.0 * CONTROL_FREQ_HZ)

# G1 hand bodies. Geoms in G1 are unnamed, so we group by body name.
LEFT_PALM_BODIES = ("left_hand_thumb_0_link",)
LEFT_FINGER_BODIES = (
    "left_hand_thumb_1_link", "left_hand_thumb_2_link",
    "left_hand_middle_0_link", "left_hand_middle_1_link",
    "left_hand_index_0_link", "left_hand_index_1_link",
)
RIGHT_PALM_BODIES = ("right_hand_thumb_0_link",)
RIGHT_FINGER_BODIES = (
    "right_hand_thumb_1_link", "right_hand_thumb_2_link",
    "right_hand_middle_0_link", "right_hand_middle_1_link",
    "right_hand_index_0_link", "right_hand_index_1_link",
)

BLOCK_GEOM_NAME = "target_block"
BLOCK_BODY_NAME = "target_block_body"


def _make_scene_xml() -> str:
    """Write wrapper XML into the G1 directory so mesh paths resolve."""
    g1_dir = MENAGERIE_PATH / "unitree_g1"
    g1_scene = g1_dir / "scene_with_hands.xml"
    if not g1_scene.exists():
        raise FileNotFoundError(
            f"scene_with_hands.xml not found at {g1_scene}"
        )

    wrapper = f"""<mujoco model="g1_lift_task">
  <include file="scene_with_hands.xml"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <worldbody>
    <body name="{BLOCK_BODY_NAME}" pos="{BLOCK_INITIAL_POS[0]} {BLOCK_INITIAL_POS[1]} {BLOCK_INITIAL_POS[2]}">
      <freejoint name="block_freejoint"/>
      <geom name="{BLOCK_GEOM_NAME}" type="box"
            size="{BLOCK_SIZE/2} {BLOCK_SIZE/2} {BLOCK_SIZE/2}"
            rgba="0.85 0.15 0.15 1"
            mass="0.2"
            friction="1.5 0.05 0.001"
            condim="4"/>
    </body>
  </worldbody>
</mujoco>
"""
    out_path = g1_dir / "_g1_lift_scene.xml"
    out_path.write_text(wrapper)
    return str(out_path)


def _find_geom_ids_by_bodies(model, body_names):
    """Return all geom IDs whose parent body is in body_names."""
    body_ids = set()
    for bn in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
        if bid >= 0:
            body_ids.add(bid)
    return [i for i in range(model.ngeom) if model.geom_bodyid[i] in body_ids]


def _find_body_id(model, candidates):
    for c in candidates:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, c)
        if bid >= 0:
            return bid
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and any(c.lower() in name.lower() for c in candidates):
            return i
    return -1


class HumanoidLiftEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_FREQ_HZ}

    def __init__(self, use_contact_bits=None,
                 render_width=640, render_height=480):
        super().__init__()

        if use_contact_bits is None:
            use_contact_bits = os.environ.get("ENV_USE_CONTACT_BITS", "1") == "1"
        self.use_contact_bits = use_contact_bits

        scene_xml = _make_scene_xml()
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 1.0 / (CONTROL_FREQ_HZ * SIM_STEPS_PER_CONTROL)

        self.block_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, BLOCK_GEOM_NAME)
        self.block_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_freejoint")
        self.block_qpos_addr = self.model.jnt_qposadr[self.block_joint_id]
        self.block_qvel_addr = self.model.jnt_dofadr[self.block_joint_id]

        self.contact_geom_ids = {
            "left_palm": _find_geom_ids_by_bodies(self.model, LEFT_PALM_BODIES),
            "left_fingers": _find_geom_ids_by_bodies(self.model, LEFT_FINGER_BODIES),
            "right_palm": _find_geom_ids_by_bodies(self.model, RIGHT_PALM_BODIES),
            "right_fingers": _find_geom_ids_by_bodies(self.model, RIGHT_FINGER_BODIES),
        }
        print("[HumanoidLiftEnv] Hand geoms found:")
        for k, v in self.contact_geom_ids.items():
            print(f"  {k}: {len(v)}")
        if all(len(v) == 0 for v in self.contact_geom_ids.values()):
            print("[HumanoidLiftEnv] WARNING: no hand geoms matched. "
                  "Contact bits will always be 0.")

        self.pelvis_body_id = _find_body_id(
            self.model, ("pelvis", "torso", "trunk", "base_link"))

        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        lo = np.where(np.isfinite(lo), lo, -1.0).astype(np.float32)
        hi = np.where(np.isfinite(hi), hi, 1.0).astype(np.float32)
        self.action_space = spaces.Box(low=lo, high=hi, dtype=np.float32)

        proprio_dim = (self.model.nq - 7) + self.model.nv
        block_dim = 3 + 4 + 3
        relative_dim = 3
        contact_dim = 4 if self.use_contact_bits else 0
        obs_dim = proprio_dim + block_dim + relative_dim + contact_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.step_count = 0
        self.lift_streak = 0
        self.prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)

        self._renderer = None
        self.render_width = render_width
        self.render_height = render_height

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        noise = self.np_random.uniform(-0.2, 0.2, size=2)
        self.data.qpos[self.block_qpos_addr:self.block_qpos_addr + 3] = [
            BLOCK_INITIAL_POS[0] + noise[0],
            BLOCK_INITIAL_POS[1] + noise[1],
            BLOCK_INITIAL_POS[2],
        ]
        self.data.qpos[self.block_qpos_addr + 3:self.block_qpos_addr + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.lift_streak = 0
        self.prev_action[:] = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        for _ in range(SIM_STEPS_PER_CONTROL):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._get_obs()
        info = self._get_info()
        reward = self._compute_reward(action)

        terminated = False
        pelvis_z = self.data.xpos[self.pelvis_body_id, 2] if self.pelvis_body_id >= 0 else 1.0
        if pelvis_z < 0.3:
            terminated = True
            reward -= 10.0
            info["fell"] = True

        block_z = self.data.qpos[self.block_qpos_addr + 2]
        if block_z > BLOCK_INITIAL_POS[2] + SUCCESS_LIFT_HEIGHT:
            self.lift_streak += 1
        else:
            self.lift_streak = 0
        if self.lift_streak >= SUCCESS_HOLD_STEPS:
            terminated = True
            reward += 100.0
            info["success"] = True

        truncated = self.step_count >= MAX_EPISODE_STEPS
        self.prev_action[:] = action
        return obs, reward, terminated, truncated, info

    def _get_info(self):
        block_pos = self.data.qpos[self.block_qpos_addr:self.block_qpos_addr + 3]
        pelvis_pos = self.data.xpos[self.pelvis_body_id] if self.pelvis_body_id >= 0 else np.zeros(3)
        return {
            "block_z": float(block_pos[2]),
            "dist_to_block": float(np.linalg.norm(pelvis_pos[:2] - block_pos[:2])),
            "lift_streak": int(self.lift_streak),
            "step_count": int(self.step_count),
            "success": False,
        }

    def _get_obs(self):
        qpos = self.data.qpos[7:].copy()
        qvel = self.data.qvel.copy()
        block_pos = self.data.qpos[self.block_qpos_addr:self.block_qpos_addr + 3].copy()
        block_quat = self.data.qpos[self.block_qpos_addr + 3:self.block_qpos_addr + 7].copy()
        block_linvel = self.data.qvel[self.block_qvel_addr:self.block_qvel_addr + 3].copy()
        pelvis_pos = self.data.xpos[self.pelvis_body_id] if self.pelvis_body_id >= 0 else np.zeros(3)
        rel_block = block_pos - pelvis_pos

        parts = [qpos, qvel, block_pos, block_quat, block_linvel, rel_block]
        if self.use_contact_bits:
            parts.append(self._get_contact_bits())
        return np.concatenate([p.astype(np.float32) for p in parts])

    def _get_contact_bits(self):
        bits = np.zeros(4, dtype=np.float32)
        keys = ["left_palm", "left_fingers", "right_palm", "right_fingers"]
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if self.block_geom_id not in (con.geom1, con.geom2):
                continue
            other = con.geom2 if con.geom1 == self.block_geom_id else con.geom1
            for j, k in enumerate(keys):
                if other in self.contact_geom_ids[k]:
                    bits[j] = 1.0
        return bits

    def _compute_reward(self, action):
        pelvis_pos = self.data.xpos[self.pelvis_body_id] if self.pelvis_body_id >= 0 else np.zeros(3)
        block_pos = self.data.qpos[self.block_qpos_addr:self.block_qpos_addr + 3]
        dist_xy = float(np.linalg.norm(pelvis_pos[:2] - block_pos[:2]))

        # Approach: dense, encourages getting hands near block (block is close)
        approach = 1.0 - np.tanh(dist_xy / 1.5)

        # Survival: heavy weight, agent must learn to stand first
        # Bonus scales with how upright the pelvis is (target ~0.79m for G1)
        pelvis_z = pelvis_pos[2]
        upright = np.clip(pelvis_z / 0.75, 0.0, 1.0)
        survive = 5.0 * upright

        # Contact bonus: agent gets credit for any hand-block contact
        contact_bonus = 0.0
        bits = self._get_contact_bits() if self.use_contact_bits else None
        if bits is not None:
            contact_bonus = 3.0 * float(bits.sum() > 0)
        else:
            for i in range(self.data.ncon):
                con = self.data.contact[i]
                if self.block_geom_id in (con.geom1, con.geom2):
                    contact_bonus = 3.0
                    break

        # Lift: scaled so it can't dominate survival
        block_z = float(block_pos[2])
        lift = max(0.0, block_z - BLOCK_INITIAL_POS[2]) * 15.0

        # Action smoothness
        action_delta = float(np.linalg.norm(action - self.prev_action))
        smoothness_penalty = -0.01 * action_delta

        return approach + survive + contact_bonus + lift + smoothness_penalty

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.render_height, width=self.render_width)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    env = HumanoidLiftEnv()
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Info: {info}")
    for _ in range(10):
        action = env.action_space.sample() * 0.0
        obs, r, term, trunc, info = env.step(action)
    print(f"After 10 steps: dist={info['dist_to_block']:.2f}, block_z={info['block_z']:.3f}")
    env.close()
    print("OK")