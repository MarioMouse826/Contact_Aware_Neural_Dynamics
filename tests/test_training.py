from __future__ import annotations

import json
from pathlib import Path

import pytest

from contact_aware_rl.config import load_experiment_config
from contact_aware_rl.experiment import evaluate_checkpoint, run_training
from contact_aware_rl.runtime import PROJECT_ROOT


def load_smoke_config_for_embodiment(embodiment: str):
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
def test_training_and_ablation_smoke(tmp_path: Path, embodiment: str) -> None:
    config = load_smoke_config_for_embodiment(embodiment)

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
    assert artifacts.latest_model_path.exists()
    assert artifacts.final_model_path.exists()
    assert artifacts.latest_model_path.name == "latest_model.zip"
    assert artifacts.final_model_path.name == "final_model.zip"
    assert (artifacts.output_dir / "monitor_history.json").exists()
    assert (artifacts.output_dir / "validation_history.json").exists()

    training_summary = json.loads(artifacts.training_summary_path.read_text())
    monitor_history = json.loads((artifacts.output_dir / "monitor_history.json").read_text())
    validation_history = json.loads((artifacts.output_dir / "validation_history.json").read_text())
    assert training_summary["training_status"] in {
        "success_checkpoint_selected",
        "no_success_checkpoint",
    }
    assert training_summary["latest_model_path"] == str(artifacts.latest_model_path)
    assert training_summary["final_model_path"] == str(artifacts.final_model_path)
    assert "transport_ready_rate" in training_summary["final_validation_metrics"]
    assert "over_goal_rate" in training_summary["final_validation_metrics"]
    assert len(monitor_history["history"]) >= 3
    assert len(validation_history["history"]) >= 3
    assert len(validation_history["history"]) == len(monitor_history["history"])

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
    assert 0.0 <= evaluation["grasp_rate"] <= 1.0
    assert 0.0 <= evaluation["lifted_grasp_rate"] <= 1.0

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
