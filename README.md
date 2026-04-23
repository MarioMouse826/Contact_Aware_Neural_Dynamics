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

For the legacy articulated arm lift-only task:

```bash
python -m contact_aware_rl.train --config configs/arm_grasp_lift.yaml --mode contact --seed 0 --num-envs 1
```

## Evaluate

```bash
python -m contact_aware_rl.evaluate --checkpoint outputs/<run-id>/best_success_model.zip --mode contact --split validation
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
python watch_ai.py --model-path outputs/<run-id>/best_success_model.zip --split validation
```

This writes an MP4 to `videos/<run-id>.mp4` and records the split/base seed in the output JSON. If a run never reaches nonzero validation success, it will not emit `best_success_model.zip`.

## Tests

```bash
python -m pytest
```
