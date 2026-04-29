"""Quick smoke test - imports and renders one frame."""
import sys

print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  MPS available: {torch.backends.mps.is_available()}")

import mujoco
print(f"MuJoCo: {mujoco.__version__}")

import stable_baselines3
print(f"SB3: {stable_baselines3.__version__}")

import gymnasium
print(f"Gymnasium: {gymnasium.__version__}")

import wandb
print(f"W&B: {wandb.__version__}")

# Try loading the G1 model
from pathlib import Path
g1_path = Path(__file__).parent.parent / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
if g1_path.exists():
    model = mujoco.MjModel.from_xml_path(str(g1_path))
    print(f"Unitree G1 loaded: {model.nq} qpos, {model.nu} actuators")
else:
    print(f"❌ G1 model not found at {g1_path}")
    sys.exit(1)

# Render one frame to confirm offscreen rendering works
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=240, width=320)
mujoco.mj_forward(model, data)
renderer.update_scene(data)
frame = renderer.render()
print(f"Rendered frame: {frame.shape}")

print("\n✅ All checks passed.")