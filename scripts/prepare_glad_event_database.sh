#!/bin/bash
# Event-local SS alignment only: no cross-event stacks or plots.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ss_python="${SS_PYTHON:-$HOME/miniconda3/envs/asdf/bin/python}"
ss_workers="${SS_WORKERS:-6}"
ss_database_run="outputs/glad_m35_event_reference"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$ss_python" - "$ss_database_run/events.json" <<'PY'
from pathlib import Path
import sys,yaml
from ss_precursor.batch import read_plan
plan=read_plan(sys.argv[1],'events-v1')
if plan['configuration']!=yaml.safe_load(Path('glad_m35_event_reference.yaml').read_text()):
    raise SystemExit('Configuration differs from frozen event plan; create a new run before editing settings.')
PY
"$ss_python" -u -m ss_precursor.event_database \
    --plan "$ss_database_run/events.json" --root "$ss_database_run/database" \
    --workers "$ss_workers"
