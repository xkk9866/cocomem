#!/bin/bash
# Resume the v20 QA comparison on the free portion of GPUs 6 and 7.
set -Eeuo pipefail
P=/data/yaokang/corrmem
cd "$P/code"
PY=$P/.venv/bin/python
exec 9>"$P/logs/q30.run.lock"
flock -n 9 || exit 2
test -f "$P/logs/Q30.done"
test ! -f "$P/logs/Q31.done" || exit 0
export CM_FIXED_CACHE=$P/cache CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=6 CM_P8=18586 CM_PT=18587
unset CM_Q8_REPLICAS
reader= student=
release() {
  local p=$1
  [ -n "$p" ] || return 0
  kill -TERM -- "-$p" 2>/dev/null || true
  wait "$p" 2>/dev/null || true
}
clean() { release "$reader"; release "$student"; }
trap clean EXIT
trap 'echo "line=$LINENO" > "$P/logs/Q31.failed"' ERR
log() { echo "[$(date -Is)] $*" | tee -a "$P/logs/q31.log"; }
scope() {
  seed=$1
  R=$P/multiseed_v20/seed$seed
  export CM_RUN_ROOT=$R CM_CACHE=$R/cache CM_STUDENT_SUFFIX=_s$seed
}
launch() {
  local gpu=$1 port=$2 path=$3 name=$4 context=$5
  shift 5
  if ss -lntH | awk '{print $4}' | grep -q ":$port$"; then log "port occupied: $port"; return 1; fi
  until nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | awk '$1>=6750{ok=1} END{exit !ok}'; do sleep 30; done
  log "loading $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_V1=0 setsid "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$path" --served-model-name "$name" --port "$port" --max-model-len "$context" \
    --gpu-memory-utilization 0.24 --max-num-seqs 4 --max-num-batched-tokens 512 --enable-chunked-prefill \
    --enforce-eager --enable-prefix-caching --disable-log-requests "$@" > "$P/logs/residual_$name.log" 2>&1 < /dev/null 9>&- &
  service=$!
}
ready() {
  local pid=$1 port=$2 name=$3
  for i in $(seq 1 500); do
    if curl -fsS --max-time 4 "localhost:$port/v1/models" 2>/dev/null | grep -Fq "\"$name\""; then return 0; fi
    kill -0 "$pid" 2>/dev/null || { tail -n 12 "$P/logs/residual_$name.log"; return 1; }
    sleep 3
  done
  return 1
}
judge() {
  launch 6 "$CM_P8" /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 8192 --kv-cache-dtype fp8 --cpu-offload-gb 1
  reader=$service
  ready "$reader" "$CM_P8" qwen25-7b
  "$PY" judge.py --model qwen25-7b --port "$CM_P8" --glob "$R/results/main3/comm20/*.json,$R/results/main3/comm20_nw/*.json" > "$R/logs/judge.log" 2>&1
  touch "$R/steps/judge.done"
  release "$reader"; reader=
}
begin_student() {
  launch 7 "$CM_PT" "$P/multiseed_v20/seed2/models/$1/merged" "$1_s2" 16384 --quantization bitsandbytes --load-format bitsandbytes
  student=$service
}
rm -f "$P/logs/Q31.failed" "$P/logs/Q31.paused"
scope 2
if [ ! -e "$R/steps/lme_main.done" ]; then begin_student tnet20; fi
scope 1
test -f "$R/steps/lme_main.done" && test -f "$R/steps/lme_nw.done"
if [ ! -e "$R/steps/judge.done" ]; then judge; fi
scope 2
for spec in 'lme_main comm20 tnet20' 'lme_nw comm20_nw tnet20nw'; do
  read -r step method variant <<< "$spec"
  [ ! -f "$R/steps/$step.done" ] || continue
  if [ -z "$reader" ]; then
    launch 6 "$CM_P8" /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 24576 --kv-cache-dtype fp8 --cpu-offload-gb 3
    reader=$service
  fi
  if [ -z "$student" ]; then begin_student "$variant"; fi
  ready "$student" "$CM_PT" "${variant}_s2"
  ready "$reader" "$CM_P8" qwen3-8b
  log "seed 2 resume $method"
  "$PY" run_corr.py --splits test200 --tag main3 --strategies "$method" --inst 4 --conc 24 > "$R/logs/$step.log" 2>&1
  touch "$R/steps/$step.done"
  release "$student"; student=
done
release "$reader"; reader=
if [ ! -f "$R/steps/judge.done" ]; then judge; fi
for s in 1 2; do
  scope "$s"
  "$PY" analyze.py --tag main3 --methods comm20,comm20_nw,oracle --ref comm20 --key label_qwen25-7b --valid_key label_qwen25-7b --json "$R/results/t30_main_qwen25-7b.json" > "$R/logs/main_table.log"
  "$PY" validate_seed_v20.py --root "$R" --seed "$s"
  touch "$R/steps/complete.done"
done
"$PY" summarize_multiseed_v20.py --root "$P/multiseed_v20" --out "$P/multiseed_v20/summary.json"
cd "$P/multiseed_v20"
find summary.json seed{0,1,2}/inputs.json seed{0,1,2}/results -type f ! -name '*.lock' -print0 | tar --null -czf "$P/r31.tgz.part" -T -
mv "$P/r31.tgz.part" "$P/r31.tgz"
touch "$P/logs/Q31.done"
log 'ALL DONE v20 witness comparison'
