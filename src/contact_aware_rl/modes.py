from __future__ import annotations

from dataclasses import dataclass

from .config import ExperimentConfig

TRAINABLE_MODES = {"baseline", "contact", "always_contact"}
EVALUATION_MODES = TRAINABLE_MODES | {"contact_ablation"}


@dataclass(frozen=True)
class ResolvedMode:
    name: str
    observation_mode: str
    contact_override: str | None
    trainable: bool
    expected_training_mode: str | None = None


def resolve_mode(mode: str) -> ResolvedMode:
    if mode == "baseline":
        return ResolvedMode(
            name=mode,
            observation_mode="baseline",
            contact_override=None,
            trainable=True,
        )
    if mode == "contact":
        return ResolvedMode(
            name=mode,
            observation_mode="contact",
            contact_override=None,
            trainable=True,
        )
    if mode == "always_contact":
        return ResolvedMode(
            name=mode,
            observation_mode="contact",
            contact_override="ones",
            trainable=True,
        )
    if mode == "contact_ablation":
        return ResolvedMode(
            name=mode,
            observation_mode="contact",
            contact_override="zeros",
            trainable=False,
            expected_training_mode="contact",
        )
    raise ValueError(f"Unsupported mode: {mode}")


def apply_mode_overrides(config: ExperimentConfig, mode: str) -> ExperimentConfig:
    resolved = resolve_mode(mode)
    updated = config.clone()
    updated.env.observation_mode = resolved.observation_mode
    updated.env.contact_override = resolved.contact_override
    return updated


def infer_mode_from_env_config(config: ExperimentConfig) -> str:
    if config.env.observation_mode == "baseline":
        return "baseline"
    if config.env.contact_override == "ones":
        return "always_contact"
    if config.env.contact_override == "zeros":
        return "contact_ablation"
    return "contact"
