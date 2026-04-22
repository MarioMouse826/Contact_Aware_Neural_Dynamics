import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces

class HumanoidLifterEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        xml_content = """
        <mujoco model="gantry_humanoid_lifter">
            <compiler angle="degree"/>
            <option gravity="0 0 -9.81" timestep="0.005"/>
            <default>
                <joint armature="0.01" damping="1"/>
                <geom friction="1 0.5 0.0001"/>
            </default>
            <worldbody>
                <light pos="0 0 3"/>
                <geom type="plane" size="3 3 0.1" rgba=".9 .9 .9 1"/>
                <body name="cube" pos="0 0.4 0.15">
                    <freejoint/>
                    <geom name="cube_geom" type="box" size="0.05 0.05 0.05" rgba="1 0 0 1" mass="0.2" friction="2 0.5 0.0001"/>
                </body>
                <body name="torso" pos="0 0 1.1">
                    <joint name="root_z" type="slide" axis="0 0 1" range="-0.5 0.5" limited="true"/>
                    <geom type="capsule" size="0.08 0.2" rgba="0.8 0.8 0.8 1"/>
                    <body name="head" pos="0 0 0.3">
                        <joint name="neck" type="hinge" axis="0 1 0" range="-45 45"/>
                        <geom type="sphere" size="0.09" rgba="0.9 0.8 0.7 1"/>
                    </body>
                    <body name="l_upper_arm" pos="-0.15 0 0.15">
                        <joint name="l_shoulder_y" type="hinge" axis="0 1 0" range="-90 90"/>
                        <joint name="l_shoulder_x" type="hinge" axis="1 0 0" range="-90 90"/>
                        <geom type="capsule" size="0.03 0.12" pos="0 0 -0.12" rgba="0.2 0.2 0.2 1"/>
                        <body name="l_lower_arm" pos="0 0 -0.24">
                            <joint name="l_elbow" type="hinge" axis="0 1 0" range="-150 0"/>
                            <geom type="capsule" size="0.025 0.12" pos="0 0 -0.12" rgba="0.3 0.3 0.3 1"/>
                            <body name="l_hand" pos="0 0 -0.24">
                                <geom name="l_hand_geom" type="box" size="0.02 0.05 0.05" rgba="0 0 1 1" friction="2 0.5 0.0001"/>
                            </body>
                        </body>
                    </body>
                    <body name="r_upper_arm" pos="0.15 0 0.15">
                        <joint name="r_shoulder_y" type="hinge" axis="0 1 0" range="-90 90"/>
                        <joint name="r_shoulder_x" type="hinge" axis="1 0 0" range="-90 90"/>
                        <geom type="capsule" size="0.03 0.12" pos="0 0 -0.12" rgba="0.2 0.2 0.2 1"/>
                        <body name="r_lower_arm" pos="0 0 -0.24">
                            <joint name="r_elbow" type="hinge" axis="0 1 0" range="-150 0"/>
                            <geom type="capsule" size="0.025 0.12" pos="0 0 -0.12" rgba="0.3 0.3 0.3 1"/>
                            <body name="r_hand" pos="0 0 -0.24">
                                <geom name="r_hand_geom" type="box" size="0.02 0.05 0.05" rgba="0 0 1 1" friction="2 0.5 0.0001"/>
                            </body>
                        </body>
                    </body>
                    <body name="l_thigh" pos="-0.08 0 -0.25">
                        <joint name="l_hip_x" type="hinge" axis="1 0 0" range="-30 30"/>
                        <joint name="l_hip_y" type="hinge" axis="0 1 0" range="-120 20"/>
                        <geom type="capsule" size="0.05 0.18" pos="0 0 -0.18" rgba="0.4 0.4 0.4 1"/>
                        <body name="l_shin" pos="0 0 -0.36">
                            <joint name="l_knee" type="hinge" axis="0 1 0" range="0 150"/>
                            <geom type="capsule" size="0.04 0.18" pos="0 0 -0.18" rgba="0.3 0.3 0.3 1"/>
                            <body name="l_foot" pos="0 0 -0.36">
                                <joint name="l_ankle" type="hinge" axis="0 1 0" range="-45 45"/>
                                <geom type="box" size="0.04 0.08 0.02" pos="0 0.04 0" rgba="0.2 0.2 0.2 1"/>
                            </body>
                        </body>
                    </body>
                    <body name="r_thigh" pos="0.08 0 -0.25">
                        <joint name="r_hip_x" type="hinge" axis="1 0 0" range="-30 30"/>
                        <joint name="r_hip_y" type="hinge" axis="0 1 0" range="-120 20"/>
                        <geom type="capsule" size="0.05 0.18" pos="0 0 -0.18" rgba="0.4 0.4 0.4 1"/>
                        <body name="r_shin" pos="0 0 -0.36">
                            <joint name="r_knee" type="hinge" axis="0 1 0" range="0 150"/>
                            <geom type="capsule" size="0.04 0.18" pos="0 0 -0.18" rgba="0.3 0.3 0.3 1"/>
                            <body name="r_foot" pos="0 0 -0.36">
                                <joint name="r_ankle" type="hinge" axis="0 1 0" range="-45 45"/>
                                <geom type="box" size="0.04 0.08 0.02" pos="0 0.04 0" rgba="0.2 0.2 0.2 1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <position name="act_neck" joint="neck" kp="100"/>
                <position name="act_ls_y" joint="l_shoulder_y" kp="100"/>
                <position name="act_ls_x" joint="l_shoulder_x" kp="100"/>
                <position name="act_le" joint="l_elbow" kp="100"/>
                <position name="act_rs_y" joint="r_shoulder_y" kp="100"/>
                <position name="act_rs_x" joint="r_shoulder_x" kp="100"/>
                <position name="act_re" joint="r_elbow" kp="100"/>
                <position name="act_lh_x" joint="l_hip_x" kp="200"/>
                <position name="act_lh_y" joint="l_hip_y" kp="200"/>
                <position name="act_lk" joint="l_knee" kp="200"/>
                <position name="act_la" joint="l_ankle" kp="200"/>
                <position name="act_rh_x" joint="r_hip_x" kp="200"/>
                <position name="act_rh_y" joint="r_hip_y" kp="200"/>
                <position name="act_rk" joint="r_knee" kp="200"/>
                <position name="act_ra" joint="r_ankle" kp="200"/>
                <position name="act_root_z" joint="root_z" kp="500"/>
            </actuator>
        </mujoco>
        """
        
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)
        
        # Configuration for Contact Signals
        self.force_thresh = 0.5
        self.l_hand_id = self.model.geom("l_hand_geom").id
        self.r_hand_id = self.model.geom("r_hand_geom").id
        self.cube_id = self.model.geom("cube_geom").id

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
        # Observation: [qpos, qvel, cube_pos, binary_contacts] = 62 dimensions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(62,), dtype=np.float32)
        self.current_step = 0

    def _get_obs(self):
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        obj_pos = self.data.body("cube").xpos
        
        # Binary Contact Detection
        c_vec = np.zeros(2, dtype=np.float32)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {contact.geom1, contact.geom2}
            
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            force_mag = np.linalg.norm(force[:3])
            
            if pair == {self.l_hand_id, self.cube_id} and force_mag > self.force_thresh:
                c_vec[0] = 1.0
            if pair == {self.r_hand_id, self.cube_id} and force_mag > self.force_thresh:
                c_vec[1] = 1.0
        
        obs = np.concatenate([qpos, qvel, obj_pos, c_vec]).astype(np.float32)
        # Ensure exact shape for SB3
        if len(obs) < 62:
            obs = np.pad(obs, (0, 62 - len(obs)))
        return obs[:62]

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        
        # Reward Logic
        left_hand_pos = self.data.body("l_hand").xpos
        right_hand_pos = self.data.body("r_hand").xpos
        cube_pos = self.data.body("cube").xpos
        
        avg_dist = (np.linalg.norm(left_hand_pos - cube_pos) + 
                    np.linalg.norm(right_hand_pos - cube_pos)) / 2.0
                    
        reach_reward = 2.0 / (1.0 + 10.0 * avg_dist)
        contact_reward = 2.0 if self.data.ncon > 0 else 0.0
        lift_reward = 50.0 if cube_pos[2] > 0.16 else 0.0
        action_penalty = -0.01 * np.sum(np.square(action))
            
        reward = reach_reward + contact_reward + lift_reward + action_penalty
        
        self.current_step += 1
        truncated = bool(self.current_step >= 200)
        terminated = False
        
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        return self._get_obs(), {}