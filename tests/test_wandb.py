from __future__ import annotations

import pytest

from contact_aware_rl.config import ExperimentConfig, LoggingConfig
from contact_aware_rl.experiment import _build_wandb_tags
from contact_aware_rl.logging_utils import start_wandb_run


class DummyRun:
    id = "dummy1234"


def test_wandb_run_does_not_set_name(monkeypatch) -> None:
    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return DummyRun()

    monkeypatch.setattr("contact_aware_rl.logging_utils.wandb.init", fake_init)

    start_wandb_run(
        config={"mode": "contact"},
        logging_config=LoggingConfig(wandb_mode="disabled"),
        job_type="train",
        tags=["contact"],
    )

    assert "name" not in captured
    assert captured["entity"] == "contact-aware-rl"
    assert captured["project"] == "contact-aware-neural-dynamics"


@pytest.mark.parametrize(
    ("embodiment", "expected_config_tag"),
    [
        ("cartesian_gripper", "config:cartesian"),
        ("arm_pinch", "config:arm"),
    ],
)
def test_build_wandb_tags_include_task_and_config_type(
    embodiment: str, expected_config_tag: str
) -> None:
    config = ExperimentConfig()
    config.env.embodiment = embodiment
    config.env.task = "grasp_lift"
    config.train.seed = 7
    config.train.num_envs = 3

    assert _build_wandb_tags(config, mode="contact") == [
        "contact",
        "task:grasp_lift",
        expected_config_tag,
        "seed:7",
        "envs:3",
    ]
