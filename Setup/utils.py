import wandb
import yaml
from pathlib import Path

WANDB_CONFIG_PATH = Path(__file__).with_name("wandb_config.yaml")

def load_wandb_config():
    with WANDB_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    missing_keys = [key for key in ("entity", "project") if key not in config]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Missing {missing} in {WANDB_CONFIG_PATH}")

    return config