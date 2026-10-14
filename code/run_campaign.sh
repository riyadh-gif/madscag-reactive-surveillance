#!/usr/bin/env bash
# One OS process per (variant, seed).
#
# TensorFlow accumulates memory across runs inside a single process:
# observed here as epoch time drifting 22ms -> 66ms/step and two
# out-of-memory kills. A fresh interpreter per run keeps each one at its
# clean-start footprint (~1.7 GB peak) and makes the campaign resumable
# at run granularity.
set -u

ROOT="${MADSCAG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MADSCAG_PY:-$ROOT/_venv_madscag/Scripts/python.exe}"
CODE="$ROOT/_madscag_run/code"
WORK="$ROOT/_madscag_run/work"
EXP="$WORK/experiments_v7"
LOG="$WORK/campaign.log"

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 TF_CPP_MIN_LOG_LEVEL=2
export MADSCAG_INPUT="$ROOT/_madscag_run/input"
export MADSCAG_WORK="$WORK"

BASE_SEEDS="42 1 2 3 7"
ABL_SEEDS="42 1 2 7"

VARIANTS_ABL="NO_ZI_B_v7 NO_PRESSURE_v7 NO_MPE_v7 NO_SPATIAL_v7 NO_ANOM_v7 NO_OBSFLAG_v7 NO_DIFF_v7 NO_GATE_v7 BETA_OBS_v7"

run_one () {
  local var="$1" seed="$2"
  if [ -f "$EXP/${var}_s${seed}/hasil.csv" ]; then
    echo "[skip] ${var}_s${seed}"
    return 0
  fi
  echo "[run ] ${var}_s${seed} $(date +%H:%M:%S)"
  "$PY" "$CODE/v7_pipeline.py" --skip-prep --no-eval \
        --filter "$var" --seeds-only "$seed" >> "$LOG" 2>&1
  local rc=$?
  if [ -f "$EXP/${var}_s${seed}/hasil.csv" ]; then
    echo "[ ok ] ${var}_s${seed}"
  else
    echo "[FAIL] ${var}_s${seed} (exit $rc) — see $LOG"
  fi
  return 0
}

for s in $BASE_SEEDS; do run_one BASE_v7 "$s"; done
for v in $VARIANTS_ABL; do
  for s in $ABL_SEEDS; do run_one "$v" "$s"; done
done

echo "=== campaign done $(date +%H:%M:%S) ==="
echo "completed runs:"
ls -d "$EXP"/*/ 2>/dev/null | while read -r d; do
  [ -f "$d/hasil.csv" ] && basename "$d"
done | tr '\n' ' '
echo
