#!/bin/bash
# Phased experiment queue (GPU1 only). Each phase runs its jobs in parallel; a supervisor restarts
# vLLM if it dies and re-runs unfinished jobs (all jobs are idempotent: LLM calls and outputs are cached).
# Progress: /data/yaokang/corrmem/logs/queue.log ; phase markers: logs/PHASE_<n>.done
P=/data/yaokang/corrmem
cd $P/code
PY=../.venv/bin/python
export CM_EMB_DEVICE=cpu
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/queue.log; }

start_server() {  # $1 model dir  $2 name  $3 memfrac
  [ -f $P/logs/vllm_cur.pid ] && kill $(cat $P/logs/vllm_cur.pid) 2>/dev/null; sleep 8
  CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=0 nohup $PY -m vllm.entrypoints.openai.api_server --model $1 \
    --served-model-name $2 --port 18431 --max-model-len 20480 --gpu-memory-utilization $3 \
    --max-num-seqs 96 --kv-cache-dtype fp8 --disable-log-requests > $P/logs/vllm_$2.log 2>&1 &
  echo $! > $P/logs/vllm_cur.pid
  for i in $(seq 1 90); do sleep 5; curl -s localhost:18431/v1/models >/dev/null && return 0; done
  return 1
}
stop_server() { [ -f $P/logs/vllm_cur.pid ] && kill $(cat $P/logs/vllm_cur.pid) 2>/dev/null; sleep 10; }

run_phase() {  # $1 phase id, $2 model, $3 name, $4 memfrac, rest: "logname|cmd"
  local ph=$1 model=$2 name=$3 mem=$4; shift 4
  [ -f $P/logs/PHASE_$ph.done ] && { log "phase $ph already done"; return; }
  for attempt in 1 2 3 4 5; do
    log "phase $ph attempt $attempt start"
    curl -s localhost:18431/v1/models | grep -q "\"$name\"" || start_server $model $name $mem || { log "server failed"; continue; }
    pids=()
    for job in "$@"; do
      n=${job%%|*}; c=${job#*|}
      bash -c "$c" >> $P/logs/$n.log 2>&1 & pids+=($!)
    done
    ok=1
    for pid in "${pids[@]}"; do wait $pid || ok=0; done
    if [ $ok = 1 ]; then touch $P/logs/PHASE_$ph.done; log "phase $ph done"; return; fi
    log "phase $ph had failures; restarting server"; stop_server
  done
}

Q8=/home/yaokang/models/Qwen3-8B-AWQ
J7=/home/yaokang/models/Qwen2.5-7B-Instruct-AWQ

# 1. memories + oracle validation + controlled strategies on dev + LoCoMo build
run_phase 1 $Q8 qwen3-8b 0.80 \
  "build_test|$PY run_build.py --splits test200 --inst 6" \
  "oracle|$PY run_sys.py --systems oracle --splits dev,test200 --tag main --inst 16" \
  "dev_corr|$PY run_corr.py --splits dev --tag dev --strategies append,coarse,self,simupdate,rebuild --inst 12" \
  "locomo_append|$PY locomo.py run --methods append"
[ -f $P/logs/PHASE_2.done ] || stop_server
# 2. counterfactual lineage attribution (prefill-only; low concurrency: full-vocab prompt logits)
run_phase 2 $Q8 qwen3-8b 0.70 \
  "cla_dev|$PY cla.py --splits dev --out $P/data/cla_dev.jsonl --notes 10 --profiles 0.5 --prof_cand 30 --conc 3"
# 3. train the Counterfactual Lineage Field (needs GPU memory: server stopped)
if [ ! -f $P/logs/PHASE_3.done ]; then stop_server; log "phase 3 train CLF"
  CUDA_VISIBLE_DEVICES=1 $PY cle.py --data $P/data/cla_dev.jsonl --out $P/models/clf >> $P/logs/clf_train.log 2>&1 && touch $P/logs/PHASE_3.done; fi
[ -f $P/logs/PHASE_4.done ] || stop_server
# 4. all methods on test200 + dev CLF variants + LoCoMo
STRAT=append,simupdate,self,coarse,clf,clf_unit,clf_nofacts,clf_overwrite,rebuild
run_phase 4 $Q8 qwen3-8b 0.75 \
  "dev_clf|CM_CLE_DEVICE=cuda $PY run_corr.py --splits dev --tag dev --strategies clf,clf_unit,clf_nofacts,clf_overwrite --inst 12" \
  "test_corr|CM_CLE_DEVICE=cuda $PY run_corr.py --splits test200 --tag main --strategies $STRAT --inst 12" \
  "test_sys|$PY run_sys.py --systems rawrag,membank,mem0 --splits test200 --tag main --inst 10" \
  "locomo_all|CM_CLE_DEVICE=cuda $PY locomo.py run --methods simupdate,coarse,self,clf,rebuild,rawrag,membank,mem0,amem"
stop_server
# 5. independent judge (different model family): Qwen2.5-7B-Instruct-AWQ
run_phase 5 $J7 qwen25-7b 0.80 \
  "judge|$PY judge.py --model qwen25-7b --glob '$P/results/*/*/*.json'"
$PY analyze.py --tag main > $P/results/table_main.txt 2>&1
$PY analyze.py --tag dev > $P/results/table_dev.txt 2>&1
log "ALL DONE"
