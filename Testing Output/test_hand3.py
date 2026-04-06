import mujoco
import mujoco.viewer
import time
import numpy as np

# (XML stays the same as your working version)
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
    for i in range(500):
        # Move the finger
        data.ctrl[0] = 0.4 if i > 100 else -0.2
        
        mujoco.mj_step(model, data)
        
        # --- THE CAND DATA POINT ---
        # 1. State: Where is the finger?
        finger_pos = data.qpos[7] # The 7th index is the slide joint
        # 2. Contact: Is it touching? (0 or 1)
        contact_active = 1.0 if data.ncon > 0 else 0.0
        
        if i % 50 == 0: # Print every 50 steps so it doesn't spam
            print(f"Step {i} | Finger Pos: {finger_pos:.2f} | Contact: {contact_active}")
            
        viewer.sync()
        time.sleep(0.01)
