from __future__ import annotations

from dataclasses import dataclass

from .config import EnvConfig, ExperimentConfig

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
    validate_mode_for_env(updated.env, mode)
    updated.env.observation_mode = resolved.observation_mode
    updated.env.contact_override = resolved.contact_override
    return updated


def trainable_modes_for_env(env_config: EnvConfig) -> tuple[str, ...]:
    if env_config.embodiment == "arm_pinch":
        return ("baseline", "contact")
    return ("baseline", "contact", "always_contact")


def evaluation_modes_for_env(env_config: EnvConfig) -> tuple[str, ...]:
    return trainable_modes_for_env(env_config) + ("contact_ablation",)


def validate_mode_for_env(env_config: EnvConfig, mode: str) -> None:
    allowed_modes = set(evaluation_modes_for_env(env_config))
    if mode not in allowed_modes:
        raise ValueError(
            f"Mode '{mode}' is not supported for embodiment '{env_config.embodiment}'. "
            f"Allowed modes: {sorted(allowed_modes)}."
        )


def infer_mode_from_env_config(config: ExperimentConfig) -> str:
    if config.env.observation_mode == "baseline":
        return "baseline"
    if config.env.contact_override == "ones":
        return "always_contact"
    if config.env.contact_override == "zeros":
        return "contact_ablation"
    return "contact"
