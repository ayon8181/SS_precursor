#!/bin/bash
# Local GLAD-M35 processing and plots; frozen GLAD event set, longer SS windows.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ss_python="${SS_PYTHON:-$HOME/miniconda3/envs/asdf/bin/python}"
ss_map_python="${SS_MAP_PYTHON:-$HOME/miniconda3/envs/geo/bin/python}"
ss_workers="${SS_WORKERS:-6}"
ss_run_root="outputs/glad_m35_10_100s_long"
if ! [[ "$ss_workers" =~ ^[1-9][0-9]*$ ]]; then
    echo 'SS_WORKERS must be a positive integer.' >&2
    exit 1
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Refuse to resume a frozen plan after configuration edits.
"$ss_python" - "$ss_run_root/events.json" <<'PY'
import sys,yaml
from pathlib import Path
from ss_precursor.batch import read_plan
plan=read_plan(sys.argv[1],'events-v1')
config=yaml.safe_load(Path('glad_m35_config.yaml').read_text())
if plan['configuration'] != config:
    raise SystemExit('Config differs from frozen plan; create a new run before changing settings.')
PY
"$ss_python" -u -m ss_precursor prepare-events \
    --plan "$ss_run_root/events.json" --root "$ss_run_root/cache" --workers "$ss_workers"
if [[ ! -f "$ss_run_root/bin_plan/plan.json" ]]; then
    "$ss_python" -u -m ss_precursor plan-bins \
        --plan "$ss_run_root/events.json" --root "$ss_run_root/cache" \
        --stack-settings stack_settings_10deg.yaml --output "$ss_run_root/bin_plan"
fi
if [[ ! -f "$ss_run_root/global_1deg/provenance.json" ]]; then
    "$ss_python" -u -m ss_precursor.distance_section \
        --plan "$ss_run_root/bin_plan/plan.json" --global-stack \
        --distance-range 100 160 --distance-step 1 --output "$ss_run_root/global_1deg"
fi
if [[ ! -f "$ss_run_root/circular_caps/provenance.json" ]]; then
    "$ss_map_python" -u -m ss_precursor.circular_stacks \
        --plan "$ss_run_root/bin_plan/plan.json" \
        --centers-script ../../SMKS/create_station.py \
        --radii-km 600 900 1200 1500 --distance-range 100 160 \
        --output "$ss_run_root/circular_caps"
fi
if [[ ! -f "$ss_run_root/vespagrams/provenance.json" ]]; then
    "$ss_python" -u -m ss_precursor.cap_vespagrams \
        --caps "$ss_run_root/circular_caps" --centers S511 S545 S579 \
        --output "$ss_run_root/vespagrams"
fi
if [[ ! -f "$ss_run_root/semblance_check/provenance.json" ]]; then
    "$ss_python" -u -m ss_precursor.semblance_check \
        --vespagrams "$ss_run_root/vespagrams" --output "$ss_run_root/semblance_check"
fi
echo "Finished: $ss_run_root"
