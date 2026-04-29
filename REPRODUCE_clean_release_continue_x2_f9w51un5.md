# Reproduce clean_release_continue_x2 f9w51un5

This note documents the exact code path and training step used for the
`f9w51un5` run under:

- `outputs/clean_release_continue_x2/f9w51un5`

This run uses the same `clean_release` code and reward logic already tracked on
the `finished_simple_hand_model` branch. The difference from the earlier
`53cisq88` run is the longer continuation budget.

## W&B run

- W&B run id: `f9w51un5`
- W&B path: `contact-aware-rl/contact-aware-neural-dynamics/f9w51un5`

## Relevant code and config

- `configs/cartesian_place_priority_motion_release_hybrid.yaml`
- `configs/clean_release.yaml`
- `src/contact_aware_rl/config.py`
- `src/contact_aware_rl/env.py`
- `src/contact_aware_rl/experiment.py`
- `src/contact_aware_rl/train.py`

## Training chain

The run was produced by the following sequence:

1. Train the motion/release hybrid stage, warm-starting from `3rfnpbbk`.
2. Train `clean_release` from the best hybrid checkpoint.
3. Continue `clean_release` once for the earlier 500000-step result.
4. Continue `clean_release` again for a 1000000-step run to produce `f9w51un5`.

The local W&B metadata for `f9w51un5` records the final stage as:

- config: `configs/clean_release.yaml`
- mode: `contact`
- seed: `0`
- num envs: `12`
- total timesteps: `1000000`
- output root: `outputs\\clean_release_continue_x2`
- init checkpoint:
  `outputs\\clean_release\\gv7u57ut\\best_model.zip`

## Final-stage command

```powershell
uv run python -m contact_aware_rl.train `
  --config configs/clean_release.yaml `
  --mode contact `
  --seed 0 `
  --num-envs 12 `
  --total-timesteps 1000000 `
  --output-root outputs\clean_release_continue_x2 `
  --wandb-mode online `
  --init-checkpoint "<path-to-clean_release-gv7u57ut-best_model.zip>"
```

## Related artifacts

- local run folder:
  `outputs/clean_release_continue_x2/f9w51un5`
- matching video:
  `videos/f9w51un5.mp4`

Artifacts and model weights for this run are expected to come from W&B and the
saved local outputs rather than from this branch.
