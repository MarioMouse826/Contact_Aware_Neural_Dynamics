"""Utilities: device resolution and W&B config loading."""
from __future__ import annotations
from pathlib import Path
import yaml
import torch

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "wandb_config.yaml"


def load_wandb_config(path=None):
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {
            "project": "contact-aware-rl",
            "entity": None,
            "device": "auto",
            "group": "humanoid-lift",
            "tags": [],
        }
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def resolve_device(requested="auto"):
    """Resolve 'auto' to the best available device.

    For SAC's small networks, CPU is often faster than MPS on Mac due to
    dispatch overhead. Set device explicitly in wandb_config.yaml to override.
    """
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cpu":
        return "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"