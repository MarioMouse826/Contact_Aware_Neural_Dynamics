from stable_baselines3 import SAC
from humanoid_env import HumanoidLifterEnv

print("🌍 Initializing Environment...")
env = HumanoidLifterEnv()

print("🧠 Initializing SAC Agent...")
# We use SAC just like you outlined in your proposal
model = SAC("MlpPolicy", env, verbose=1)

print("🚀 Starting Training (This will take a while!)")
# In your proposal, you set a budget of 500K steps. 
# Let's start with 10,000 just to make sure it works without crashing.
model.learn(total_timesteps=50000, log_interval=1, progress_bar=True)

print("💾 Saving the trained brain...")
model.save("sac_humanoid_lifter")

print("✅ Done!")