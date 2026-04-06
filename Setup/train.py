from stable_baselines3 import SAC
from humanoid_env import HumanoidLifterEnv

print("🌍 Initializing Environment...")
env = HumanoidLifterEnv()

print("🧠 Initializing SAC Agent...")
# Soft-Actor Critic Agent is outlined like in the proposal
model = SAC("MlpPolicy", env, verbose=1)

print("🚀 Starting Training (This will take a while!)")
# We have a budget of 500K steps total 
model.learn(total_timesteps=50000, log_interval=1, progress_bar=True)

print("💾 Saving the trained brain...")
model.save("sac_humanoid_lifter")

print("✅ Done!")
