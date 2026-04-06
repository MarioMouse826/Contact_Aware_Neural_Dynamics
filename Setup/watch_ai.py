import time
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from humanoid_env import HumanoidLifterEnv

print("🌍 Loading Environment...")
env = HumanoidLifterEnv()

print("🧠 Loading Trained Brain...")
# Load the brain we just saved!
model = SAC.load("sac_humanoid_lifter")

# Get the initial starting state
obs, info = env.reset()

print("🎥 Opening Viewer... Watch your AI!")
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        # 1. The AI looks at the observation and predicts the best action
        # deterministic=True means it takes the "best" action instead of exploring
        action, _states = model.predict(obs, deterministic=True)
        
        # 2. We apply that action to the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. Update the 3D visualizer
        viewer.sync()
        
        # 4. Add a slight delay so we can watch it in real-time
        # (Physics timestep is 0.005, and we step 5 times per AI action = 0.025s)
        time.sleep(0.025)
        
        # 5. If the 200-step horizon is reached, reset the simulation!
        if terminated or truncated:
            obs, info = env.reset()