import mujoco
import mujoco.viewer
import time
import numpy as np

xml_content = """
<mujoco>
    <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
        
        <body name="cube" pos="0 0 0.2">
            <joint type="free"/>
            <geom type="box" size="0.05 0.05 0.05" rgba="1 0 0 1" mass="0.1"/>
        </body>

        <body name="finger" pos="-0.2 0 0.05">
            <joint name="finger_x" type="slide" axis="1 0 0"/>
            <geom type="capsule" size="0.02 0.05" rgba="0 0 1 1" euler="0 90 0"/>
        </body>
    </worldbody>
    <actuator>
        <position name="move_finger" joint="finger_x" kp="100"/>
    </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_content)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(1000):
        # Every 200 steps, tell the finger to move toward the cube
        if i > 200:
            data.ctrl[0] = 0.3 # Move finger to position 0.3
        
        mujoco.mj_step(model, data)
        
        # LOGGING: This is what the CAND 'Encoder' sees
        if data.ncon > 0:
            print(f"💥 CONTACT DETECTED at step {i}!")
            
        viewer.sync()
        time.sleep(0.01)

# 3. Launch the 3D Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 10: # Runs for 10 seconds
        step_start = time.time()

        # Physics step
        mujoco.mj_step(model, data)

        # Sync the 3D window
        viewer.sync()

        # Slow it down to real-time speed
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)

print("✅ Simulation Finished.")
