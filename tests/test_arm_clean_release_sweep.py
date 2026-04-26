from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from contact_aware_rl.config import ExperimentConfig


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_arm_clean_release_sweep.py"
)


def load_sweep_module():
    spec = importlib.util.spec_from_file_location("run_arm_clean_release_sweep", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_required_checkpoint_fails_without_fallback(tmp_path: Path) -> None:
    sweep = load_sweep_module()

    missing = tmp_path / "missing.zip"
    try:
        sweep._resolve_required_checkpoint(str(missing))
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing checkpoint should fail")


def test_success_target_uses_strict_success_rate_only() -> None:
    sweep = load_sweep_module()

    staged_only = {
        "success_rate": 0.0,
        "transport_ready_rate": 1.0,
        "over_goal_rate": 1.0,
        "placement_rate": 1.0,
    }
    assert not sweep._meets_success_target(staged_only, target_success_rate=0.10)
    assert sweep._meets_success_target({"success_rate": 0.10}, target_success_rate=0.10)


def test_target_hit_only_stops_when_explicitly_requested() -> None:
    sweep = load_sweep_module()

    assert not sweep._should_stop_after_stage(target_met=True, stop_on_target=False)
    assert sweep._should_stop_after_stage(target_met=True, stop_on_target=True)
    assert not sweep._should_stop_after_stage(target_met=False, stop_on_target=True)


def test_apply_recipe_adds_wandb_sweep_tags(tmp_path: Path) -> None:
    sweep = load_sweep_module()
    config = ExperimentConfig()
    config.env.embodiment = "arm_pinch"
    config.env.arm_control_mode = "ee_delta"

    updated = sweep._apply_recipe(
        config,
        sweep.SweepRecipe("formula_nominal"),
        stage="warm",
        output_root=tmp_path / "run",
    )

    assert "arm-clean-release-sweep" in updated.logging.wandb_tags
    assert "source:iconic-haze-72" in updated.logging.wandb_tags
    assert "recipe:formula_nominal" in updated.logging.wandb_tags
    assert "stage:warm" in updated.logging.wandb_tags
    assert "control:ee_delta" in updated.logging.wandb_tags


def test_run_stage_passes_init_checkpoint_to_training(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sweep = load_sweep_module()
    captured = {}

    def fake_run_training(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(sweep, "run_training", fake_run_training)
    config = ExperimentConfig()
    config.env.embodiment = "arm_pinch"
    config.env.task = "pick_place_ab"
    args = argparse.Namespace(
        num_envs=2,
        total_timesteps=1234,
        warm_timesteps=None,
        continue_timesteps=500_000,
        wandb_mode="disabled",
    )

    sweep._run_stage(
        base_config=config,
        recipe=sweep.SweepRecipe("formula_nominal", seed=7),
        stage="warm",
        init_checkpoint="/tmp/iconic-haze-72.zip",
        output_dir=tmp_path,
        args=args,
    )

    assert captured["kwargs"]["init_checkpoint"] == "/tmp/iconic-haze-72.zip"
    assert captured["kwargs"]["seed"] == 7
    assert captured["kwargs"]["num_envs"] == 2
    assert captured["kwargs"]["total_timesteps"] == 1234
    assert captured["kwargs"]["wandb_mode"] == "disabled"
