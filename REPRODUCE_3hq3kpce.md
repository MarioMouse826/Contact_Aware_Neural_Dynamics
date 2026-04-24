# Reproduce 3hq3kpce

This branch packages the code, config, trained artifacts, and reference video for the `3hq3kpce` run.

## What is included

- Exact long-run config: `configs/cartesian_place_priority_release_corridor_long.yaml`
- Saved run folder: `outputs/cartesian_place_priority_release_corridor_long/3hq3kpce`
- Reference video: `videos/3hq3kpce.mp4`

## Run the trained model

```powershell
Set-Location "C:\Users\12890\ml project\Contact_Aware_Neural_Dynamics_first_success"

uv run python watch_ai.py `
  --model-path "outputs\cartesian_place_priority_release_corridor_long\3hq3kpce\best_model.zip" `
  --split validation `
  --episodes 1 `
  --output-video "videos\3hq3kpce_rerender.mp4"
```

## Retrain with the same setup

```powershell
Set-Location "C:\Users\12890\ml project\Contact_Aware_Neural_Dynamics_first_success"

uv sync

uv run python -m contact_aware_rl.train `
  --config configs/cartesian_place_priority_release_corridor_long.yaml `
  --mode contact `
  --seed 0 `
  --wandb-mode online
```

This should reproduce a similar result, but exact RL outcomes can still vary across machines and runs.
