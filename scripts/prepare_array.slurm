#!/usr/bin/env bash
#SBATCH --job-name=ss-sy-events
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/ss-events-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ss-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
"${SS_PY:-python3}" -u -m ss_precursor prepare-events \
    --plan "${1:?Event plan JSON required}" --root "${2:?Cache root required}" \
    --task-index "${SLURM_ARRAY_TASK_ID:?Submit as a job array}"
