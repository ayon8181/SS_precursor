#!/bin/bash
#SBATCH --job-name=ss-sy
#SBATCH --account=EAR21003
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=logs/ss-vista-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"

# Arguments: Vista Python executable, quoted ASDF glob, output root, optional config.
ss_python="${1:?Pass the Python executable from your Vista environment}"
ss_inputs="${2:?Pass a quoted ASDF glob}"
ss_run_root="${3:-outputs/vista_sy}"
ss_config="${4:-sy_config.yaml}"
ss_workers="${SS_WORKERS:-16}"
if ! [[ "$ss_workers" =~ ^[1-9][0-9]*$ ]] || (( ss_workers > ${SLURM_CPUS_PER_TASK:-72} )); then
    echo 'SS_WORKERS must be a positive integer no greater than the allocated CPU count.' >&2
    exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ss-mpl-${SLURM_JOB_ID}"
mkdir -p "$ss_run_root"
echo "Module list"
if type module >/dev/null 2>&1; then module list; fi
echo "Nodes: ${SLURM_JOB_NODELIST:-unknown}"
echo "Python: $ss_python; workers: $ss_workers"
echo "Starting SS stacking workflow: $(date)"

# All catalog inspection, processing, and stacking run on the compute node.
# Existing plans freeze settings; choose a new output root to change inputs/config.
if [[ ! -f "$ss_run_root/events.json" ]]; then
    "$ss_python" -u -m ss_precursor plan-events "$ss_inputs" \
        --config "$ss_config" --network SY --output "$ss_run_root/events.json"
fi
"$ss_python" -u -m ss_precursor prepare-events \
    --plan "$ss_run_root/events.json" --root "$ss_run_root/cache" --workers "$ss_workers"

if [[ ! -f "$ss_run_root/bin_plan/plan.json" ]]; then
    "$ss_python" -u -m ss_precursor plan-bins \
        --plan "$ss_run_root/events.json" --root "$ss_run_root/cache" \
        --output "$ss_run_root/bin_plan"
fi
"$ss_python" -u -m ss_precursor stack-bins \
    --plan "$ss_run_root/bin_plan/plan.json" --output "$ss_run_root/stacks" \
    --workers "$ss_workers" --plots
echo "Finished SS stacking workflow: $(date)"
