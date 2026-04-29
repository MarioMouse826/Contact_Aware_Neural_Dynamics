from __future__ import annotations

import numpy as np
import pytest

from contact_aware_rl.config import load_experiment_config
from contact_aware_rl.experiment import _make_vector_env
from contact_aware_rl.modes import apply_mode_overrides
from contact_aware_rl.runtime import PROJECT_ROOT


def _load_vector_env_config(embodiment: str):
    smoke_config = load_experiment_config(PROJECT_ROOT / "configs" / "smoke.yaml")
    if embodiment != "arm_pinch":
        smoke_config.env.embodiment = embodiment
        return smoke_config

    arm_config = load_experiment_config(PROJECT_ROOT / "configs" / "arm_box.yaml")
    arm_config.env.max_episode_steps = smoke_config.env.max_episode_steps
    arm_config.train = smoke_config.train
    arm_config.eval = smoke_config.eval
    arm_config.logging = smoke_config.logging
    return arm_config


@pytest.mark.parametrize("embodiment", ["cartesian_gripper", "arm_pinch"])
def test_parallel_vector_env_step(embodiment: str) -> None:
    config = _load_vector_env_config(embodiment)
    config = apply_mode_overrides(config, "contact")
    config.train.num_envs = 2

    env = _make_vector_env(config)
    try:
        obs = env.reset()
        assert obs.shape[0] == 2

        action_dim = int(env.action_space.shape[0])
        next_obs, rewards, dones, infos = env.step(
            np.zeros((2, action_dim), dtype=np.float32)
        )
        assert next_obs.shape[0] == 2
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
    finally:
        env.close()
