#!/usr/bin/env bash
set -euo pipefail

# Fair no-contact reruns for the final contact-aware comparisons.
#
# These are observation ablations only: --mode baseline removes the binary
# contact bits from policy observations while preserving simulator contact
# physics, reward shaping, SAC architecture, seed, eval splits, and task setup.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WANDB_MODE="${WANDB_MODE:-online}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-12}"

ARM_WARM_CHECKPOINT="${ARM_WARM_CHECKPOINT:-outputs/baselines/fresh-firebrand-75/warm/kz9qnjn7/best_model.zip}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

run_training() {
  "$PYTHON_BIN" -m contact_aware_rl.train "$@"
}

require_file configs/baselines/dashing_water_66_exact.yaml
require_file configs/baselines/fresh_firebrand_75_no_plateau.yaml
require_file "$ARM_WARM_CHECKPOINT"

echo "== Cartesian exact baseline for dashing-water-66 / f9w51un5 =="
run_training \
  --config configs/baselines/dashing_water_66_exact.yaml \
  --mode baseline \
  --seed "$SEED" \
  --num-envs "$NUM_ENVS" \
  --total-timesteps 1000000 \
  --output-root outputs/baselines/dashing-water-66/cartesian_exact \
  --wandb-mode "$WANDB_MODE"

echo "== Arm no-plateau continue baseline for fresh-firebrand-75 / fbusrjsg =="
run_training \
  --config configs/baselines/fresh_firebrand_75_no_plateau.yaml \
  --mode baseline \
  --seed "$SEED" \
  --num-envs "$NUM_ENVS" \
  --total-timesteps 500000 \
  --output-root outputs/baselines/fresh-firebrand-75/continue_no_plateau \
  --wandb-mode "$WANDB_MODE" \
  --init-checkpoint "$ARM_WARM_CHECKPOINT"

cat <<SUMMARY

Fair no-contact baseline reruns submitted.

Expected W&B comparisons:
  Cartesian contact: dashing-water-66 / f9w51un5
  Cartesian baseline: newest run under outputs/baselines/dashing-water-66/cartesian_exact

  Arm contact: fresh-firebrand-75 / fbusrjsg
  Arm baseline: newest run under outputs/baselines/fresh-firebrand-75/continue_no_plateau
SUMMARY
