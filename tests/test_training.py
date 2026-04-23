from __future__ import annotations

import json
from pathlib import Path

import pytest

from contact_aware_rl.config import load_experiment_config
from contact_aware_rl.experiment import evaluate_checkpoint, run_training
from contact_aware_rl.runtime import PROJECT_ROOT


@pytest.mark.parametrize("embodiment", ["cartesian_gripper", "arm_pinch"])
def test_training_and_ablation_smoke(tmp_path: Path, embodiment: str) -> None:
    config = load_experiment_config(PROJECT_ROOT / "configs" / "smoke.yaml")
    config.env.embodiment = embodiment

    artifacts = run_training(
        config,
        mode="contact",
        seed=0,
        num_envs=1,
        output_root=str(tmp_path / embodiment / "outputs"),
        wandb_mode="disabled",
    )

    assert artifacts.config_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.training_summary_path.exists()
    assert artifacts.final_model_path.exists()
    assert (artifacts.output_dir / "monitor_history.json").exists()
    assert (artifacts.output_dir / "validation_history.json").exists()

    training_summary = json.loads(artifacts.training_summary_path.read_text())
    assert training_summary["training_status"] in {
        "success_checkpoint_selected",
        "no_success_checkpoint",
    }

    evaluation = evaluate_checkpoint(
        artifacts.final_model_path,
        mode="contact",
        split="custom",
        episodes=2,
        base_seed=123,
    )
    assert evaluation["requested_mode"] == "contact"
    assert evaluation["split"] == "custom"
    assert evaluation["base_seed"] == 123
    assert 0.0 <= evaluation["success_rate"] <= 1.0

    if artifacts.best_success_model_path.exists():
        ablation = evaluate_checkpoint(
            artifacts.best_success_model_path,
            mode="contact_ablation",
            split="validation",
            episodes=2,
        )
        assert ablation["requested_mode"] == "contact_ablation"
        assert 0.0 <= ablation["success_rate"] <= 1.0
    else:
        assert training_summary["training_status"] == "no_success_checkpoint"
