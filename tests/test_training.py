from __future__ import annotations

from pathlib import Path

from contact_aware_rl.config import load_experiment_config
from contact_aware_rl.experiment import evaluate_checkpoint, run_training
from contact_aware_rl.runtime import PROJECT_ROOT


def test_training_and_ablation_smoke(tmp_path: Path) -> None:
    config = load_experiment_config(PROJECT_ROOT / "configs" / "smoke.yaml")

    artifacts = run_training(
        config,
        mode="contact",
        seed=0,
        num_envs=1,
        output_root=str(tmp_path / "outputs"),
        wandb_mode="disabled",
    )

    assert artifacts.config_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.training_summary_path.exists()
    assert artifacts.best_model_path.exists()
    assert artifacts.final_model_path.exists()

    ablation = evaluate_checkpoint(
        artifacts.best_model_path,
        mode="contact_ablation",
        episodes=2,
    )
    assert ablation["requested_mode"] == "contact_ablation"
    assert 0.0 <= ablation["success_rate"] <= 1.0
