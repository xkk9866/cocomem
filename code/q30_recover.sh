#!/bin/bash
# Resume a failed witnessed student while preserving an already-running no-witness training job.
set -Eeuo pipefail
P=/data/yaokang/corrmem
cd "$P/code"
PY=../.venv/bin/python
exec 8>"$P/logs/q30.recovery.lock"
flock -n 8 || exit 1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$P/logs/q30.log"; }
fail() { log "RECOVERY_FAILED line=$1 status=$2"; echo "RECOVERY_FAILED line=$1 status=$2" > "$P/logs/Q30.failed"; exit 1; }
trap 'fail "$LINENO" "$?"' ERR
export CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
train_one() {
  local gpu=$1 variant=$2
  if [ ! -e "$P/models/$variant/train_meta.json" ]; then
    log "recovery train $variant GPU=$gpu tok_budget=4096 effective_batch=16"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" commutator.py train --out "$P/models/$variant" --max_len 3072 --tok_budget 4096 >> "$P/logs/q30_train_$variant.log" 2>&1
  fi
  test -s "$P/models/$variant/train_meta.json"
  test -s "$P/models/$variant/merged/config.json"
  log "trained $variant GPU=$gpu (recovery)"
}
rm -f "$P/logs/Q30.failed"
log "recover training; retaining existing tnet20nw PID=${1:-none}"
( train_one 6 tnet20; train_one 6 tnet20nv ) & left=$!
(
  old_pid=${1:-0}
  while [ "$old_pid" != 0 ] && [ -r "/proc/$old_pid/cmdline" ] && grep -aq 'models/tnet20nw' "/proc/$old_pid/cmdline"; do sleep 30; done
  train_one 7 tnet20nw
  train_one 7 tnet20nr
) & right=$!
wait "$left"
wait "$right"
log 'recovery training complete; resume q30 evaluation'
flock -u 8
exec 8>&-
exec bash q30.sh
