from __future__ import annotations

import numpy as np

from contact_aware_rl.config import load_experiment_config
from contact_aware_rl.experiment import _make_vector_env
from contact_aware_rl.modes import apply_mode_overrides
from contact_aware_rl.runtime import PROJECT_ROOT


def test_parallel_vector_env_step() -> None:
    config = load_experiment_config(PROJECT_ROOT / "configs" / "smoke.yaml")
    config = apply_mode_overrides(config, "contact")
    config.train.num_envs = 2

    env = _make_vector_env(config)
    try:
        obs = env.reset()
        assert obs.shape[0] == 2

        next_obs, rewards, dones, infos = env.step(np.zeros((2, 4), dtype=np.float32))
        assert next_obs.shape[0] == 2
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
    finally:
        env.close()
