import mujoco
import mujoco.viewer
import time

# 1. Define the 'World' 
xml = """
<mujoco>
    <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
        <body name="cube" pos="0 0 0.2">
            <joint type="free"/>
            <geom type="box" size="0.05 0.05 0.05" rgba="1 0 0 1"/>
        </body>
        <body name="finger" pos="0.08 0 0.1">
            <geom type="sphere" size="0.04" rgba="0 0 1 1"/>
        </body>
    </worldbody>
</mujoco>
"""

# 2. Initialize the Physics
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("🚀 Launching 3D Viewer...")

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
