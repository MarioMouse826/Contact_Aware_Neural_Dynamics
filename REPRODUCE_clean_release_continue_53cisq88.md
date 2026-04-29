# Reproduce clean_release_continue 53cisq88

This branch contains the code and configs used for the `53cisq88` run under
`outputs/clean_release_continue`.

## Relevant configs

- `configs/cartesian_place_priority_motion_release_hybrid.yaml`
- `configs/clean_release.yaml`

## Training flow

The `53cisq88` run is a continuation of the `clean_release` stage, which itself
builds on the motion/release hybrid setup.

1. Train the motion/release hybrid model, starting from the smoother
   `3rfnpbbk` checkpoint from W&B.
2. Train `clean_release`, initializing from the best checkpoint of the hybrid run.
3. Continue `clean_release` for 500000 timesteps to get the `53cisq88`-style result.

## Commands

Train the hybrid stage:

```powershell
uv run python -m contact_aware_rl.train `
  --config configs/cartesian_place_priority_motion_release_hybrid.yaml `
  --mode contact `
  --seed 0 `
  --num-envs 12 `
  --total-timesteps 400000 `
  --output-root outputs\cartesian_place_priority_motion_release_hybrid `
  --wandb-mode online `
  --init-checkpoint "<path-to-3rfnpbbk-best_model.zip>"
```

Train `clean_release`:

```powershell
uv run python -m contact_aware_rl.train `
  --config configs/clean_release.yaml `
  --mode contact `
  --seed 0 `
  --num-envs 12 `
  --total-timesteps 400000 `
  --output-root outputs\clean_release `
  --wandb-mode online `
  --init-checkpoint "<path-to-hybrid-best_model.zip>"
```

Continue `clean_release` for the 500000-step run:

```powershell
uv run python -m contact_aware_rl.train `
  --config configs/clean_release.yaml `
  --mode contact `
  --seed 0 `
  --num-envs 12 `
  --total-timesteps 500000 `
  --output-root outputs\clean_release_continue `
  --wandb-mode online `
  --init-checkpoint "<path-to-clean_release-best_model.zip>"
```

Artifacts and logs for these runs are expected to come from W&B rather than this branch.
