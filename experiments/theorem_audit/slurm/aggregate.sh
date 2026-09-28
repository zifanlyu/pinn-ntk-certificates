#!/usr/bin/env bash
#SBATCH --job-name=ntk-aggregate
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=experiments/theorem_audit/logs/%x_%j.out
#SBATCH --error=experiments/theorem_audit/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
CONFIG="${CONFIG:-experiments/theorem_audit/configs/paper.json}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PYTHON_COMMAND <<< "$PYTHON_BIN"
elif command -v uv >/dev/null 2>&1; then
  PYTHON_COMMAND=(uv run python)
elif [[ -x .venv/bin/python ]]; then
  PYTHON_COMMAND=(.venv/bin/python)
else
  PYTHON_COMMAND=(python)
fi

"${PYTHON_COMMAND[@]}" experiments/theorem_audit/run.py aggregate \
  --config "$CONFIG"

