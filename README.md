# Contact-Aware Neural Dynamics

This repository contains a fresh implementation of the final project experiments for contact-aware reinforcement learning. The code uses a custom MuJoCo grasp-and-lift environment and compares SAC with and without binary finger contact observations, following the motivation from `Contact-Aware Neural Dynamics.pdf` and the experiment design in `proposal.md`.

## Experiments

- `baseline`: SAC without contact bits in the observation.
- `contact`: SAC with the true binary contact bits appended to the observation.
- `always_contact`: SAC with the contact bits forced to `1`.
- `contact_ablation`: evaluation-only mode that zeros out contact bits for a trained `contact` policy.

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
.venv/bin/python -m contact_aware_rl.train --mode contact --seed 0 --num-envs 1
```

## Evaluate

```bash
.venv/bin/python -m contact_aware_rl.evaluate --checkpoint outputs/<run-id>/best_model.zip --mode contact --episodes 20
```

For the proposal ablation:

```bash
.venv/bin/python -m contact_aware_rl.evaluate --checkpoint outputs/<run-id>/best_model.zip --mode contact_ablation --episodes 20
```

## Run The Proposal Suite

```bash
.venv/bin/python -m contact_aware_rl.sweep --suite proposal --seeds 0 1 2 --num-envs 1
```

## Tests

```bash
.venv/bin/python -m pytest
```
