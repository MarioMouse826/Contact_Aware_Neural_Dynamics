from __future__ import annotations

from contact_aware_rl.config import LoggingConfig
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
