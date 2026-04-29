from __future__ import annotations

from pathlib import Path

import pytest

from contact_aware_rl.config import ExperimentConfig, LoggingConfig
from contact_aware_rl.experiment import _build_wandb_tags
from contact_aware_rl.logging_utils import save_wandb_files, start_wandb_run


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


def test_save_wandb_files_uploads_existing_files_with_run_root_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = []
    best_model = tmp_path / "best_model.zip"
    latest_model = tmp_path / "latest_model.zip"
    best_model.write_bytes(b"best")
    latest_model.write_bytes(b"latest")

    monkeypatch.setattr("contact_aware_rl.logging_utils.wandb.run", DummyRun())

    def fake_save(glob_str, *, base_path, policy):
        captured.append((glob_str, base_path, policy))
        return [glob_str]

    monkeypatch.setattr("contact_aware_rl.logging_utils.wandb.save", fake_save)

    saved = save_wandb_files(
        [best_model, latest_model, tmp_path / "missing.zip"],
        base_path=tmp_path,
    )

    assert saved == [str(best_model), str(latest_model)]
    assert captured == [
        (str(best_model), str(tmp_path), "now"),
        (str(latest_model), str(tmp_path), "now"),
    ]


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


def test_build_wandb_tags_appends_configured_tags() -> None:
    config = ExperimentConfig()
    config.logging.wandb_tags = ["arm-closed-loop-sweep", "variant:ee_nominal"]

    assert _build_wandb_tags(config, mode="contact")[-2:] == [
        "arm-closed-loop-sweep",
        "variant:ee_nominal",
    ]
