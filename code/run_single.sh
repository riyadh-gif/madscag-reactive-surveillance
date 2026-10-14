#!/usr/bin/env bash
set -u
ROOT="${MADSCAG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${MADSCAG_PY:-$ROOT/_venv_madscag/Scripts/python.exe}"
CODE="$ROOT/_madscag_run/code"
WORK="$ROOT/_madscag_run/work"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 TF_CPP_MIN_LOG_LEVEL=2
export MADSCAG_INPUT="$ROOT/_madscag_run/input" MADSCAG_WORK="$WORK"
for s in 42 1 2 7; do
  if [ -f "$WORK/experiments_v7/BL_SINGLE_s${s}/hasil.csv" ]; then
    echo "[skip] BL_SINGLE_s$s"; continue; fi
  echo "[run ] BL_SINGLE_s$s $(date +%H:%M:%S)"
  "$PY" "$CODE/baselines.py" BL_SINGLE "$s" >> "$WORK/baselines.log" 2>&1
  [ -f "$WORK/experiments_v7/BL_SINGLE_s${s}/hasil.csv" ] && echo "[ ok ] BL_SINGLE_s$s" || echo "[FAIL] BL_SINGLE_s$s"
done
echo "=== BL_SINGLE done ==="
