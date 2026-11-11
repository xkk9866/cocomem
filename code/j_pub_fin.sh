#!/bin/bash
# Finish A-Mem on the two running 8B servers, rerun its LoCoMo memories with the current implementation,
# judge with Qwen2.5-7B on any card with room (GPUs 0-7), and export the published-system table data.
set -Euo pipefail
P=/data/yaokang/corrmem
cd $P/code
PY=../.venv/bin/python
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CM_EMB_DEVICE=cpu
J7=/data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ
log() { echo "[$(date '+%H:%M:%S')] $*"; }

free_gpus() {   # gpus with at least $1 MiB free, most free first
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
    awk -F', ' -v need="$1" '{f=$3-$2; if (f>=need) print f, $1, $3}' | sort -rn
}
up() { curl -sf localhost:$1/v1/models >/dev/null 2>&1; }
serve_any() {   # serve_any PORT MODEL NAME : first card that can hold it
  local pt=$1 m=$2 n=$3
  up $pt && return 0
  while read -r f g t; do
    local F; F=$(python3 -c "print(round(max(0.20, min(0.40, ($f-2500)/$t)), 3))")
    log "try $n on gpu $g free $f frac $F port $pt"
    if bash start_srv.sh $g $pt $m $n $F 8192; then return 0; fi
    bash stop_vllm.sh $pt || true
  done < <(free_gpus 11000)
  return 1
}

REP=18622,18623
mkdir -p $P/results/locomo_amem_old $P/states/locomo_amem_old
mv $P/results/locomo/amem/*.json $P/results/locomo_amem_old/ 2>/dev/null || true
mv $P/states/locomo_amem/*.json $P/states/locomo_amem_old/ 2>/dev/null || true

CM_P8=18622 CM_Q8_REPLICAS=$REP $PY run_sys.py --systems amem --splits test200 --tag main3 --inst 13 --conc 32 \
  --shard 1:2 --reverse > $P/logs/pub_am1r.log 2>&1 &
W1=$!
CM_P8=18623 CM_Q8_REPLICAS=18623,18622 $PY locomo.py run --methods amem > $P/logs/pub_locomo_amem.log 2>&1 &
W2=$!
log "workers $W1 $W2"

until serve_any 18620 $J7 qwen25-7b; do sleep 60; done
$PY judge.py --model qwen25-7b --port 18620 --glob "$P/results/main3/amem/*.json" > $P/logs/pub_judge_am.log 2>&1
log "first judge pass exit $?"

wait $W1; log "reverse worker exit $?"
wait $W2; log "locomo amem exit $?"
while [ "$(ls $P/results/main3/amem | grep -c 'json$')" -lt 201 ]; do
  log "amem $(ls $P/results/main3/amem | grep -c 'json$')/201"; sleep 60
done
for d in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null || true)
  case "$cmd" in *"--systems amem"*) kill "${d##*/}" 2>/dev/null || true ;; esac
done 2>/dev/null
touch $P/logs/AMEM.done
bash stop_vllm.sh 18622 || true
bash stop_vllm.sh 18623 || true

$PY judge.py --model qwen25-7b --port 18620 --glob \
  "$P/results/main3/amem/*.json,$P/results/locomo/amem/*.json,$P/results/main3/ga/*.json,$P/results/main3/readagent/*.json,$P/results/locomo/ga/*.json,$P/results/locomo/readagent/*.json" \
  > $P/logs/pub_judge_final.log 2>&1
log "final judge exit $?"
bash stop_vllm.sh 18620 || true
$PY export_pub.py > $P/logs/pub_export.log 2>&1
log PUB_DONE
touch $P/logs/PUB.done
