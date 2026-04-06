from pathlib import Path

import torch
import yaml

WANDB_CONFIG_PATH = Path(__file__).with_name("wandb_config.yaml")


def load_wandb_config():
    with WANDB_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    missing_keys = [key for key in ("entity", "project") if key not in config]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Missing {missing} in {WANDB_CONFIG_PATH}")

    config["device"] = normalize_device_name(config.get("device", "auto"))
    return config


def normalize_device_name(device_name):
    normalized = str(device_name).strip().lower()
    if normalized == "gpu":
        normalized = "cuda"

    valid_devices = {"auto", "cpu", "mps", "cuda"}
    if normalized not in valid_devices:
        valid_list = ", ".join(sorted(valid_devices | {"gpu"}))
        raise ValueError(
            f"Unsupported device '{device_name}' in {WANDB_CONFIG_PATH}. "
            f"Use one of: {valid_list}."
        )

    return normalized


def resolve_device(requested_device):
    requested_device = normalize_device_name(requested_device)

    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _is_mps_available():
            return "mps"
        return "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "Configured device 'cuda' is not available on this machine. "
            "Use 'auto' or switch the config to 'cpu' or 'mps'."
        )

    if requested_device == "mps" and not _is_mps_available():
        raise ValueError(
            "Configured device 'mps' is not available on this machine. "
            "Use 'auto' or switch the config to 'cpu' or 'cuda'."
        )

    return requested_device


def _is_mps_available():
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    return bool(mps_backend.is_built() and mps_backend.is_available())
