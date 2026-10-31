#!/usr/bin/env bash
# Usage: scripts/kaggle_launch.sh <configA> <configB> [extra train.py overrides...]
# Runs two configs in parallel, one per GPU. With a single GPU they run one after the other.
# Env: OUT (/kaggle/working), RESUME_ROOT (a dir holding <run>/ckpt_*.pt), MAX_HOURS (11.5)
set -u
A=$1; B=$2; shift 2
OUT=${OUT:-/kaggle/working}
MAX_HOURS=${MAX_HOURS:-11.5}
LOGS=$OUT/logs
EXTRA=("$@")
mkdir -p "$LOGS"
GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

run() { # gpu config hours
    local resume=()
    if [ -n "${RESUME_ROOT:-}" ] && [ -d "$RESUME_ROOT/$2" ]; then
        resume=(--resume_from "$RESUME_ROOT/$2")
    fi
    CUDA_VISIBLE_DEVICES=$1 nohup python train.py --config "config/train/$2.yaml" --max_hours "$3" "${resume[@]}" \
        train.output_root="$OUT" "${EXTRA[@]}" >> "$LOGS/$2.log" 2>&1
}

if [ "$GPUS" -ge 2 ]; then
    run 0 "$A" "$MAX_HOURS" & P0=$!
    run 1 "$B" "$MAX_HOURS" & P1=$!
    echo "[launch] $A on GPU 0 (pid $P0), $B on GPU 1 (pid $P1); logs in $LOGS"
    wait $P0; S0=$?
    wait $P1; S1=$?
else
    HALF=$(python -c "print($MAX_HOURS / 2)")
    echo "[launch] $GPUS GPU visible: running $A then $B, $HALF h each; logs in $LOGS"
    run 0 "$A" "$HALF"; S0=$?
    run 0 "$B" "$HALF"; S1=$?
fi
echo "[launch] $A exit $S0, $B exit $S1"
[ "$S0" -eq 0 ] && [ "$S1" -eq 0 ]
