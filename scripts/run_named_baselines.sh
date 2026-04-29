#!/usr/bin/env bash
set -euo pipefail

# Baseline ablation runner for the named W&B contact-aware runs:
# - absurd-bee-45: Cartesian pick/place run 3rfnpbbk
# - fresh-firebrand-75: arm clean-release continuation run fbusrjsg
#
# These commands use --mode baseline, which removes the binary contact bits from
# policy observations while keeping the same task, reward, seeds, and eval setup.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WANDB_MODE="${WANDB_MODE:-online}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-12}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/baselines}"

run_training() {
  "$PYTHON_BIN" -m contact_aware_rl.train "$@"
}

latest_run_dir() {
  local root="$1"
  find "$root" -mindepth 1 -maxdepth 1 -type d -print0 \
    | xargs -0 ls -td \
    | head -n 1
}

require_checkpoint() {
  local checkpoint="$1"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing required checkpoint: $checkpoint" >&2
    exit 1
  fi
}

echo "== Cartesian baseline for W&B run absurd-bee-45 / 3rfnpbbk =="
run_training \
  --config outputs/cartesian_place_priority/3rfnpbbk/config.yaml \
  --mode baseline \
  --seed "$SEED" \
  --num-envs "$NUM_ENVS" \
  --total-timesteps 500000 \
  --output-root "$OUTPUT_ROOT/absurd-bee-45/cartesian" \
  --wandb-mode "$WANDB_MODE"

cartesian_run_dir="$(latest_run_dir "$OUTPUT_ROOT/absurd-bee-45/cartesian")"
echo "Cartesian baseline output: $cartesian_run_dir"

echo "== Arm source baseline matching upstream contact run 6mgbe4ps =="
run_training \
  --config outputs/arm_closed_loop_sweep/20260425_010319/runs/ee_long_seed0/6mgbe4ps/config.yaml \
  --mode baseline \
  --seed "$SEED" \
  --num-envs "$NUM_ENVS" \
  --total-timesteps 1000000 \
  --output-root "$OUTPUT_ROOT/fresh-firebrand-75/source" \
  --wandb-mode "$WANDB_MODE"

source_run_dir="$(latest_run_dir "$OUTPUT_ROOT/fresh-firebrand-75/source")"
source_checkpoint="$source_run_dir/best_model.zip"
require_checkpoint "$source_checkpoint"
echo "Arm source baseline output: $source_run_dir"

echo "== Arm clean-release warm baseline matching warm contact run 1jnwqpn3 =="
run_training \
  --config outputs/arm_clean_release_sweep/20260426_001702/runs/formula_nominal/warm/1jnwqpn3/config.yaml \
  --mode baseline \
  --seed "$SEED" \
  --num-envs "$NUM_ENVS" \
  --total-timesteps 400000 \
  --output-root "$OUTPUT_ROOT/fresh-firebrand-75/warm" \
  --wandb-mode "$WANDB_MODE" \
  --init-checkpoint "$source_checkpoint"

warm_run_dir="$(latest_run_dir "$OUTPUT_ROOT/fresh-firebrand-75/warm")"
warm_checkpoint="$warm_run_dir/best_model.zip"
require_checkpoint "$warm_checkpoint"
echo "Arm warm baseline output: $warm_run_dir"

echo "== Arm clean-release continue baseline for W&B run fresh-firebrand-75 / fbusrjsg =="
run_training \
  --config outputs/arm_clean_release_sweep/20260426_001702/runs/formula_nominal/continue/fbusrjsg/config.yaml \
  --mode baseline \
  --seed "$SEED" \
  --num-envs "$NUM_ENVS" \
  --total-timesteps 500000 \
  --output-root "$OUTPUT_ROOT/fresh-firebrand-75/continue" \
  --wandb-mode "$WANDB_MODE" \
  --init-checkpoint "$warm_checkpoint"

continue_run_dir="$(latest_run_dir "$OUTPUT_ROOT/fresh-firebrand-75/continue")"
echo "Arm continue baseline output: $continue_run_dir"

cat <<SUMMARY

Baseline runs complete.

Cartesian baseline:
  $cartesian_run_dir

Arm baseline chain:
  source:   $source_run_dir
  warm:     $warm_run_dir
  continue: $continue_run_dir

Use the final training_summary.json files in those directories against:
  absurd-bee-45 / 3rfnpbbk
  fresh-firebrand-75 / fbusrjsg
SUMMARY
