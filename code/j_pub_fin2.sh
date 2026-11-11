#!/bin/bash
# Wait for the running A-Mem workers, free their servers, then judge on any card with room (GPUs 0-7) and export.
set -Euo pipefail
P=/data/yaokang/corrmem
cd $P/code
PY=../.venv/bin/python
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CM_EMB_DEVICE=cpu
J7=/data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ
log() { echo "[$(date '+%H:%M:%S')] $*"; }
running() {   # running PATTERN : any process whose command line contains PATTERN
  local d cmd
  for d in /proc/[0-9]*; do
    cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
    case "$cmd" in *"$1"*) [ "${d##*/}" != "$$" ] && return 0 ;; esac
  done 2>/dev/null
  return 1
}
nfiles() { ls "$1" 2>/dev/null | grep -c 'json$' || true; }
up() { curl -sf localhost:$1/v1/models >/dev/null 2>&1; }
free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
    awk -F', ' -v need="$1" '{f=$3-$2; if (f>=need) print f, $1, $3}' | sort -rn
}
serve_judge() {
  up 18620 && return 0
  local f g t F
  while read -r f g t; do
    F=$(python3 -c "print(round(max(0.34, min(0.40, ($f-1500)/$t)), 3))")
    log "judge try gpu $g free $f frac $F"
    bash start_srv.sh $g 18620 $J7 qwen25-7b $F 8192 && return 0
    bash stop_vllm.sh 18620 || true
  done < <(free_gpus 9500)
  return 1
}

while running "locomo.py run --methods amem"; do sleep 60; done
log "locomo amem finished: $(nfiles $P/results/locomo/amem)/10"
if [ "$(nfiles $P/results/locomo/amem)" -lt 10 ]; then
  CM_P8=18623 CM_Q8_REPLICAS=18623,18622 $PY locomo.py run --methods amem >> $P/logs/pub_locomo_amem.log 2>&1
  log "locomo amem retry: $(nfiles $P/results/locomo/amem)/10"
fi
while [ "$(nfiles $P/results/main3/amem)" -lt 201 ]; do
  running "--systems amem" || { log "no A-Mem worker left at $(nfiles $P/results/main3/amem)"; \
    CM_P8=18622 CM_Q8_REPLICAS=18622,18623 $PY run_sys.py --systems amem --splits test200 --tag main3 --inst 13 --conc 32 >> $P/logs/pub_am1r.log 2>&1; }
  sleep 60
done
log "amem 201/201"
for d in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  case "$cmd" in *"--systems amem"*) kill "${d##*/}" 2>/dev/null || true ;; esac
done 2>/dev/null
touch $P/logs/AMEM.done
bash stop_vllm.sh 18622 || true
bash stop_vllm.sh 18623 || true

until serve_judge; do sleep 60; done
$PY judge.py --model qwen25-7b --port 18620 --glob \
  "$P/results/main3/amem/*.json,$P/results/locomo/amem/*.json,$P/results/main3/ga/*.json,$P/results/main3/readagent/*.json,$P/results/locomo/ga/*.json,$P/results/locomo/readagent/*.json" \
  > $P/logs/pub_judge_final.log 2>&1
log "judge exit $?"
bash stop_vllm.sh 18620 || true
$PY export_pub.py > $P/logs/pub_export.log 2>&1
log "export: $(tail -n 1 $P/logs/pub_export.log)"
log PUB_DONE
touch $P/logs/PUB.done
