# Contact_Aware_Neural_Dynamics
MuJoCo simulation experimenting with Contact Aware Neural Dynamics

## What Is W&B?

Weights & Biases, usually called W&B, is a tool for tracking machine learning experiments. In this project, each training run can automatically log:

- the hyperparameters used for training
- reward curves and other training metrics
- the code snapshot associated with the run

This makes it easier to compare runs across teammates and answer questions like:

- Which learning rate worked better?
- Did a code change improve reward?
- Which model checkpoint came from which experiment?

Some basic W&B terms:

- `entity`: your W&B workspace or team name
- `project`: the shared bucket where runs are stored
- `run`: one execution of `python Setup/train.py`

## Using uv

We prefer `uv` as the package manager for this project. `uv` creates and manages the local virtual environment and installs the dependencies from `pyproject.toml`.

The main commands you need are:

- `uv sync`: install the project dependencies into the local environment
- `uv run python Setup/train.py`: run the training script inside that environment
- `uv run python Setup/watch_ai.py`: run the viewer inside that environment

If you add a new package to the project later, use:

- `uv add <package-name>`

## Weights & Biases

Edit `Setup/wandb_config.yaml` with your W&B workspace info:

```yaml
entity: contact-aware-rl
project: my-awesome-project
device: auto
```

The `device` setting controls where PyTorch and Stable-Baselines3 run:

- `auto`: prefer `cuda`, then `mps`, then `cpu`
- `cuda`: require an NVIDIA GPU
- `mps`: require Apple Silicon MPS
- `cpu`: force CPU only
- `gpu`: accepted as an alias for `cuda`

Then install dependencies and run training:

```bash
uv sync
uv run python Setup/train.py
```

`Setup/train.py` will call `wandb.login()` automatically and start a run in the configured project. It also writes Stable-Baselines3 TensorBoard logs and lets W&B sync them automatically. You do not need to use the TensorBoard UI, but the `tensorboard` Python package should be installed. Device selection is read from `Setup/wandb_config.yaml`, and the requested/resolved device is logged to W&B for each run. The training hyperparameters live near the top of `Setup/train.py`, so if you want to compare runs you can just change those values and run the script again.

If you set `device` to `cuda` or `mps`, that backend must actually be available in your local PyTorch install. If not, the script will fail with a clear error. Only `device: auto` falls back automatically.

## First-Time Setup

If you have never used W&B before:

1. Create an account at `wandb.ai`.
2. Ask for the correct team or workspace name if you are logging to a shared class/team account.
3. Put the correct `entity` and `project` in `Setup/wandb_config.yaml`.
4. Pick a device in `Setup/wandb_config.yaml`. In most cases, use `device: auto`.
5. Run `uv sync` and then `uv run python Setup/train.py`.
6. The first time, W&B may ask you to paste an API key. After that, it is usually saved on your machine.

After training starts, you should see a link in the terminal to the W&B run page.

## Watch A Trained Policy

```bash
uv run python Setup/watch_ai.py --model-path sac_humanoid_lifter
```

`Setup/watch_ai.py` uses the same `device` setting as training when loading the saved model.
