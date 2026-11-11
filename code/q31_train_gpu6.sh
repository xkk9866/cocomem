#!/bin/bash
# Use the available GPU for the already-planned seed 2 training.
set -Eeuo pipefail
P=/data/yaokang/corrmem
cd "$P/code"
PY=$P/.venv/bin/python
exec 9>"$P/logs/q30.run.lock"
flock -n 9 || { echo 'Another experiment queue owns the lock'; exit 1; }
test -f "$P/logs/Q30.done"
L=$P/logs/q31_train_gpu6.log
log() { echo "[$(date -Is)] $*" >> "$L"; }
trap 'log "FAILED line=$LINENO status=$?"' ERR
nvidia-smi -i 6 --query-gpu=memory.free --format=csv,noheader,nounits | awk '{exit !($1>=22000)}'
export CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=6 CM_FIXED_CACHE=$P/cache
unset CM_Q8_REPLICAS
R=$P/multiseed_v20/seed2
"$PY" prepare_multiseed_v20.py --seed 2 >> "$L"
export CM_RUN_ROOT=$R CM_CACHE=$R/cache CM_STUDENT_SUFFIX=_s2
for variant in tnet20 tnet20nw tnet20nv tnet20nr; do
  if [ ! -f "$R/models/$variant/train_meta.json" ]; then
    log "seed=2 train=$variant GPU=6"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" commutator.py train \
      --out "$R/models/$variant" --seed 2 --max_len 3072 --tok_budget 4096 \
      > "$R/logs/train_$variant.log" 2>&1
  fi
  test -s "$R/models/$variant/merged/config.json"
  cp "$R/models/$variant/train_meta.json" "$R/results/train_meta_$variant.json"
  log "seed=2 trained=$variant"
done
log 'seed 2 training complete; reconnecting evaluation queue'
unset CM_RUN_ROOT CM_CACHE CM_STUDENT_SUFFIX
flock -u 9
exec 9>&-
exec bash q31_wait.sh
