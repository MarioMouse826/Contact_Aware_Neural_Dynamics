import mujoco
import mujoco.viewer
import time
import numpy as np

# A fully parameterized bipedal humanoid with a free-floating base
# Note that <freejoint> is used for the torso, allowing it to move freely in space, while the limbs are connected with hinge joints for realistic articulation.
# The cube is placed in front of the humanoid to test interaction, and the actuators are set up to control the joints for potential lifting actions.
# Here is the problem: 500K steps is nowhere near enough to train a humanoid to lift a cube. The humanoid doesn't know how to stand up let alone lift a cube.
# We need to train this for much longer, and we also need to implement a reward function that encourages the humanoid to lift the cube. 
xml_content = """
<mujoco model="humanoid_lifter">
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
            <freejoint name="root"/>
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
    </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_content)
data = mujoco.MjData(model)

print("🚀 Launching Full Humanoid...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(2000):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)

print("✅ Simulation Finished.")
