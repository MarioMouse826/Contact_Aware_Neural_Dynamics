import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces

class HumanoidLifterEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 1. The full Gantry Humanoid XML
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
                    <geom type="box" size="0.05 0.05 0.05" rgba="1 0 0 1" mass="0.2" friction="2 0.5 0.0001"/>
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
                                <geom type="box" size="0.02 0.05 0.05" rgba="0 0 1 1" friction="2 0.5 0.0001"/>
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
                                <geom type="box" size="0.02 0.05 0.05" rgba="0 0 1 1" friction="2 0.5 0.0001"/>
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
        
        # 2. CRUCIAL FIX: "self." MUST be here so the environment remembers them
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)
        
        # 3. Setup the AI Action and Observation Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32)

    def _get_obs(self):
        """
        Extract the state as follows
        s_t = [q, q_dot, x_obj, R_obj, x_k, delta_x] + [c_t]
        """
        # q: joint positions, q_dot: joint velocities
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        
        # x_obj: box position
        obj_pos = self.data.body("cube").xpos
        
        # c_t: The Binary Contact Signal! 
        # Are the hands touching the box?
        left_contact = 0.0
        right_contact = 0.0
        if self.data.ncon > 0:
            # Simplification: if contacts exist, we flag it. 
            # (In reality, we'd check the specific geom IDs for the hands and the box)
            left_contact = 1.0 
            right_contact = 1.0
            
        contact_vector = np.array([left_contact, right_contact], dtype=np.float32)
        
        # Concatenate everything into one giant 1D array for the neural network
        obs = np.concatenate([qpos, qvel, obj_pos, contact_vector]).astype(np.float32)
        
        # Pad with zeros to ensure it exactly matches the shape=(60,) we defined
        obs = np.pad(obs, (0, 60 - len(obs))) 
        return obs

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        
        # =======================================================
        # REWARD SHAPING 
        # =======================================================
        
        # 1. Action Regularization: Dialed way down (from 0.5 to 0.01)
        # We just want to discourage flailing, not paralyze it completely.
        action_penalty = -0.01 * np.sum(np.square(action))
        
        # 2. Reach Reward: Positive Exponential Curve
        # Instead of a negative punishment, we make it a positive number that 
        # gets MASSIVE as the hands get closer to the box.
        left_hand_pos = self.data.body("l_hand").xpos
        right_hand_pos = self.data.body("r_hand").xpos
        cube_pos = self.data.body("cube").xpos
        
        avg_dist = (np.linalg.norm(left_hand_pos - cube_pos) + 
                    np.linalg.norm(right_hand_pos - cube_pos)) / 2.0
                    
        # This creates a "gravity well" of reward pulling the hands to the box
        reach_reward = 2.0 / (1.0 + 10.0 * avg_dist)
        
        # 3. Contact Reward
        contact_reward = 2.0 if self.data.ncon > 0 else 0.0
            
        # 4. Lift Reward
        lift_reward = 50.0 if cube_pos[2] > 0.16 else 0.0
            
        reward = reach_reward + contact_reward + lift_reward + action_penalty
        
        # Tick the clock
        self.current_step += 1
        truncated = bool(self.current_step >= 200)
        terminated = False
        
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Puts the robot and box back to the start position."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data) # Ensure all derived quantities are correct

        #Start the clock!
        self.current_step = 0
        
        # Optionally randomize the box position slightly here to force the AI to generalize
        
        return self._get_obs(), {}
