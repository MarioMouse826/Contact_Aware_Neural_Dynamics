# Contact-Aware Neural Dynamics

This repository contains a fresh implementation of the final project experiments for contact-aware reinforcement learning. The code uses custom MuJoCo manipulation environments and compares SAC with and without binary finger contact observations, following the motivation from `Contact-Aware Neural Dynamics.pdf` and the experiment design in `proposal.md`.

The default Cartesian task is now an explicit tabletop pick-and-place objective: start with the cube at point A, grasp it, move it to point B, place it back on the table, release it, and let it settle. The legacy grasp-and-lift task remains available through a separate config.

## Experiments

- `baseline`: SAC without contact bits in the observation.
- `contact`: SAC with the true binary contact bits appended to the observation.
- `always_contact`: SAC with the contact bits forced to `1` for the Cartesian gripper task only.
- `contact_ablation`: evaluation-only mode that zeros out contact bits for a trained `contact` policy.

## Embodiments

- `cartesian_gripper`: the original slide-joint tabletop gripper. `configs/default.yaml` runs the new `pick_place_ab` task by default. Use `configs/cartesian_grasp_lift.yaml` for the legacy lift-only task.
- `arm_pinch`: a fixed-base articulated arm with four revolute arm joints and a two-finger pinch hand. `configs/arm_box.yaml` now also runs the `pick_place_ab` task. Use `configs/arm_grasp_lift.yaml` for the legacy lift-only task.

The articulated arm supports `baseline`, `contact`, and `contact_ablation`. It does not support `always_contact`.

Pick-place resets are seeded-randomized around the configured start marker for both embodiments. Evaluation now records monitor and validation summaries at every `eval_freq`, not only when the monitor tuple improves.

All runs log to Weights and Biases under:

- `entity=contact-aware-rl`
- `project=contact-aware-neural-dynamics`

The code never assigns a custom W&B run name, so W&B keeps its default random naming.

## Setup

```bash
uv sync
```

## Train

```bash
python -m contact_aware_rl.train --mode contact --seed 0 --num-envs 1
```

For the legacy Cartesian lift-only task:

```bash
python -m contact_aware_rl.train --config configs/cartesian_grasp_lift.yaml --mode contact --seed 0 --num-envs 1
```

For the articulated arm pick-and-place task:

```bash
python -m contact_aware_rl.train --config configs/arm_box.yaml --mode contact --seed 0 --num-envs 1
```

`configs/arm_box.yaml` is the intended arm pick-place entry point. It uses an arm home pose near the pick-place start marker plus stronger pre-lift grasp/lift shaping than the generic defaults.

The corrected articulated-arm place-priority run uses closed-loop end-effector control:

```bash
python -m contact_aware_rl.train --config configs/arm_ee_place_priority.yaml --mode contact
```

The older joint-space arm reproduction config remains available for diagnosis:

```bash
python -m contact_aware_rl.train --config configs/arm_place_priority.yaml --mode contact
```

Compare against the staged pick-place metrics from `3rfnpbbk`: transport-ready rate around `0.64`, over-goal rate around `0.64`, placement rate around `0.53`, and mean best goal distance around `0.13`.

For the overnight closed-loop arm sweep:

```bash
python scripts/run_arm_closed_loop_sweep.py --wandb-mode online
```

For a short local smoke run of the sweep machinery:

```bash
python scripts/run_arm_closed_loop_sweep.py --wandb-mode disabled --max-runs 1 --num-envs 1 --total-timesteps 1000 --allow-incomplete
```

For the arm clean-release warm-start sweep from the `iconic-haze-72` checkpoint,
choose the config that matches the checkpoint action space:

```bash
python scripts/run_arm_clean_release_sweep.py \
  --config configs/arm_clean_release_joint.yaml \
  --init-checkpoint <path-to-iconic-haze-72-model.zip> \
  --wandb-mode online
```

Use `configs/arm_clean_release_ee.yaml` only for checkpoints trained with
`arm_control_mode: ee_delta`. The sweep requires the checkpoint path and runs
every default recipe through both `warm` and `continue` stages. Add
`--stop-on-target` only when you want a shorter scout run that exits after the
first strict validation success.

Single-run warm starts are also supported:

```bash
python -m contact_aware_rl.train \
  --config configs/arm_clean_release_joint.yaml \
  --mode contact \
  --init-checkpoint <path-to-iconic-haze-72-model.zip>
```

For the stabilized Cartesian release-corridor experiment:

```bash
python -m contact_aware_rl.train --config configs/cartesian_place_priority_release_stabilized.yaml --mode contact --seed 0 --num-envs 1
```

This runs a warmup stage to select `best_success_model.zip`, then resumes from that policy with lower learning rate and fixed entropy for late-training stabilization.

For the next Cartesian release-corridor improvement sweep against the `glamorous-plant-49` baseline, use:

```bash
python -m contact_aware_rl.train --config configs/cartesian_place_priority_release_glamorous_repro_selector.yaml --mode contact
python -m contact_aware_rl.train --config configs/cartesian_place_priority_release_post_release_stability.yaml --mode contact
python -m contact_aware_rl.train --config configs/cartesian_place_priority_release_long_horizon.yaml --mode contact
python -m contact_aware_rl.train --config configs/cartesian_place_priority_release_combined_stability_horizon.yaml --mode contact
```

These variants keep the original release-corridor task/evaluation criteria, use `num_envs=12` and one million timesteps, and select `best_model.zip` by task-completion score instead of strict success rate alone.

For the legacy articulated arm lift-only task:

```bash
python -m contact_aware_rl.train --config configs/arm_grasp_lift.yaml --mode contact --seed 0 --num-envs 1
```

## Evaluate

```bash
python -m contact_aware_rl.evaluate --checkpoint outputs/<run-id>/best_model.zip --mode contact --split validation
```

For the proposal ablation:

```bash
python -m contact_aware_rl.evaluate --checkpoint outputs/<run-id>/best_success_model.zip --mode contact_ablation --split validation
```

## Run The Proposal Suite

```bash
python -m contact_aware_rl.sweep --suite proposal --seeds 0 1 2 --num-envs 1
```

## Record A Video

```bash
python watch_ai.py --model-path outputs/<run-id>/best_model.zip --split validation
```

This writes an MP4 to `videos/<run-id>.mp4` and records the split/base seed in the output JSON. Every run emits `best_model.zip` and `latest_model.zip`; `best_success_model.zip` is still only written after nonzero validation success.

## Tests

```bash
python -m pytest
```
