#!/bin/bash
# Fixed data pipeline, independent training seeds 1/2; seed 0 is q30.
set -Eeuo pipefail
P=/data/yaokang/corrmem
cd "$P/code"
PY=$P/.venv/bin/python
exec 9>"$P/logs/q30.run.lock"
flock -n 9 || { echo 'q30 still owns the GPUs'; exit 1; }
test -f "$P/logs/Q30.done" || { echo 'q30 is not complete'; exit 1; }
L=$P/logs/q31.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$L"; }
fail() { log "FAILED line=$1 status=$2"; echo "line=$1 status=$2" > "$P/logs/Q31.failed"; exit 1; }
trap 'fail "$LINENO" "$?"' ERR
rm -f "$P/logs/Q31.failed"
export CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=6 CM_FIXED_CACHE=$P/cache
unset CM_Q8_REPLICAS
Q8=/home/yaokang/models/Qwen3-8B-AWQ
METHODS=comm20,comm20_nw
test -s "$P/multiseed_v20/seed0/results/t30_main_qwen25-7b.json"
step() {
  local name=$1; shift
  if [ ! -e "$R/steps/$name.done" ]; then
    log "seed=$seed start $name"
    "$@" > "$R/logs/$name.log" 2>&1
    touch "$R/steps/$name.done"
    log "seed=$seed done $name"
  fi
}
select_seed() {
  seed=$1
  R=$P/multiseed_v20/seed$seed
  export CM_RUN_ROOT=$R CM_CACHE=$R/cache CM_STUDENT_SUFFIX=_s$seed
}
reader_up=0
for seed_id in 1 2; do
  select_seed "$seed_id"
  "$PY" prepare_multiseed_v20.py --seed "$seed" >> "$L"
  for variant in tnet20 tnet20nw; do
    if [ "$variant" = tnet20 ]; then method=comm20; name=lme_main; else method=comm20_nw; name=lme_nw; fi
    if [ -f "$R/steps/$name.done" ]; then continue; fi
    if [ "$reader_up" = 0 ]; then
      bash start_q31_reader.sh >> "$L"
      reader_up=1
    fi
    test -s "$R/models/$variant/train_meta.json"
    bash start_tnet.sh "$R/models/$variant/merged" 0.36 "${variant}_s$seed" 18432 6 >> "$L"
    step "$name" env CM_PT=18432 "$PY" run_corr.py --splits test200 --tag main3 --strategies "$method" --inst 8 --conc 64
    bash stop_vllm.sh 18432
  done
done
if [ "$reader_up" = 1 ]; then bash stop_vllm.sh 18431; fi
bash start_srv.sh 6 18431 /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.85 >> "$L"
for seed_id in 1 2; do
  select_seed "$seed_id"
  if [ -f "$R/steps/complete.done" ]; then continue; fi
  G="$R/results/main3/comm20/*.json,$R/results/main3/comm20_nw/*.json"
  step judge "$PY" judge.py --model qwen25-7b --port 18431 --glob "$G"
  step main_table "$PY" analyze.py --tag main3 --methods "$METHODS,oracle" --ref comm20 --key label_qwen25-7b --valid_key label_qwen25-7b --json "$R/results/t30_main_qwen25-7b.json"
  "$PY" validate_seed_v20.py --root "$R" --seed "$seed"
  touch "$R/steps/complete.done"
  log "seed=$seed complete"
done
bash stop_vllm.sh 18431
"$PY" summarize_multiseed_v20.py --root "$P/multiseed_v20" --out "$P/multiseed_v20/summary.json"
cd "$P/multiseed_v20"
find summary.json seed0/inputs.json seed0/results seed1/inputs.json seed1/results seed2/inputs.json seed2/results -type f -print0 | tar --null -czf "$P/r31.tgz" -T -
touch "$P/logs/Q31.done"
log 'ALL DONE seeds 1,2'
