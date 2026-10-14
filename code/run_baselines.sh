#!/usr/bin/env bash
# One OS process per (baseline, seed), same reason as run_campaign.sh:
# TensorFlow does not release memory cleanly across repeated model builds.
set -u

ROOT="${MADSCAG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MADSCAG_PY:-$ROOT/_venv_madscag/Scripts/python.exe}"
CODE="$ROOT/_madscag_run/code"
WORK="$ROOT/_madscag_run/work"
EXP="$WORK/experiments_v7"
LOG="$WORK/baselines.log"

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 TF_CPP_MIN_LOG_LEVEL=2
export MADSCAG_INPUT="$ROOT/_madscag_run/input"
export MADSCAG_WORK="$WORK"

SEEDS="42 1 2 7"
NEURAL="BL_GRUD BL_LSTM BL_TCN"

# deterministic, seed-independent: force re-run so they pick up the new
# metric columns and write preds npz
echo "[run ] constants $(date +%H:%M:%S)"
rm -rf "$EXP/BL_CLIM_s0" "$EXP/BL_ZERO_s0" "$EXP/BL_PERSIST_s0"
"$PY" "$CODE/baselines.py" BL_CLIM,BL_ZERO,BL_PERSIST >> "$LOG" 2>&1
for n in BL_CLIM BL_ZERO BL_PERSIST; do
  [ -f "$EXP/${n}_s0/hasil.csv" ] && echo "[ ok ] ${n}_s0" || echo "[FAIL] ${n}_s0"
done

run_one () {
  local kind="$1" seed="$2"
  if [ -f "$EXP/${kind}_s${seed}/hasil.csv" ]; then
    echo "[skip] ${kind}_s${seed}"; return 0
  fi
  echo "[run ] ${kind}_s${seed} $(date +%H:%M:%S)"
  "$PY" "$CODE/baselines.py" "$kind" "$seed" >> "$LOG" 2>&1
  local rc=$?
  if [ -f "$EXP/${kind}_s${seed}/hasil.csv" ]; then
    echo "[ ok ] ${kind}_s${seed}"
  else
    echo "[FAIL] ${kind}_s${seed} (exit $rc) — see $LOG"
  fi
  return 0
}

# XGBoost two-stage first: it is the decisive comparison and the cheapest
for s in $SEEDS; do run_one BL_XGB2 "$s"; done
for k in $NEURAL; do
  for s in $SEEDS; do run_one "$k" "$s"; done
done

echo "=== baselines done $(date +%H:%M:%S) ==="
ls -d "$EXP"/BL_*/ 2>/dev/null | while read -r d; do
  [ -f "$d/hasil.csv" ] && basename "$d"
done | tr '\n' ' '
echo
