#!/bin/bash
# Sourced by the v21 job scripts started by sched21.py.  Environment from the scheduler:
#   J_GPU  = GPU index, or "g1,g2" (g1 serves the 8B compressor, g2 the 4B student)
#   J_FREE = free MiB on g1 at launch, J_FREE2 = free MiB on g2 (= J_FREE when one GPU)
#   J_PORT = port base   J_JOB = job id
# The GPUs are shared with other users: every server takes a fraction of the card computed from the memory that was
# free at launch, with a margin, and runs in eager mode with a bounded batch so that its footprint is predictable.
set -Eeuo pipefail
P=${CM_ROOT:-/data/yaokang/corrmem}; cd $P/code; PY=../.venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# The single-rank engine of a server still owns an NCCL process group whose watchdog ends the process when its
# heartbeat is late by 480 s.  On this host a server that waits for requests while memory is being reclaimed can be
# stalled that long (11-11 01:07: the 8B server of the main evaluation was ended this way after 47 minutes of work).
# The watchdog monitors collectives between ranks; there is one rank, so it is switched off for every server.
export TORCH_NCCL_ENABLE_MONITORING=0
G=${J_GPU:-0}; G1=${G%%,*}; G2=${G##*,}
PORT=${J_PORT:-18450}; FREE=${J_FREE:-24000}; FREE2=${J_FREE2:-$FREE}; J_JOB=${J_JOB:-job}
P8=$PORT; P4=$((PORT+1)); P8B=$((PORT+2))
TOTAL=24564
frac() { python3 -c "import sys; print(max(0.05, min(0.92, round(float(sys.argv[1])/$TOTAL, 3))))" "$1"; }
mib() { python3 -c "import sys; print(int(float(sys.argv[1])*$TOTAL))" "$1"; }
# memory budgets (MiB) of one server, derived from the free memory of its GPU at launch.  A quarter of the free
# memory is left as headroom: neighbours' processes grow during the minutes a server needs to start on the loaded
# host, and a server that is squeezed out while allocating its cache dies (observed: 13.9 GB free at launch, OOM
# six minutes later)
budget8() { local f=$1; if [ "$f" -ge 20000 ]; then echo $((f*80/100)); else echo $((f*75/100)); fi; }
budget4() { local f=$1; if [ "$f" -ge 20000 ]; then echo 6000; else local b=$((f*75/100)); [ "$b" -gt 5200 ] && b=5200; echo $b; fi; }
# context window the 8B server gets for a budget: int4 weights 5.85 GB + fp8 KV cache (74 KB/token) + activations;
# the full 24k window needs about 8.6 GB; below that the window is reduced and callers defer over-long requests
# (llm.ContextOverflow); below 7.6 GB the server cannot hold a 16k window
maxlen8() { local b=$1; if [ "$b" -ge 8600 ]; then echo 24576; elif [ "$b" -ge 7900 ]; then echo 20480; else echo 16384; fi; }
MIN8=7600
Q8=/home/yaokang/models/Qwen3-8B-AWQ
# the /data disk is shared and often saturated (1-2 MB/s for one reader): model weights are read from the NVMe copies
# under /home/yaokang/corrmem_nvme/models when they exist, otherwise from their original location
NV=/home/yaokang/corrmem_nvme
Q25=$NV/models/Qwen2.5-7B-Instruct-AWQ; [ -e $Q25/config.json ] || Q25=/data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ
LLAMA=$NV/models/Llama-3.1-8B-Instruct-AWQ; [ -e $LLAMA/config.json ] || LLAMA=/data/yaokang/models/Llama-3.1-8B-Instruct-AWQ
log() { local m="[$(date '+%m-%d %H:%M:%S')] [$J_JOB] $*"; echo "$m"; echo "$m" >> $P/logs/sched21_jobs.log; }
up() { curl -s localhost:$1/v1/models | grep -q "\"$2\""; }
JSTART=$P/logs/.jstart_$J_JOB; touch $JSTART
# A server that runs out of KV cache preempts requests, and preempted requests have come back corrupted on these
# servers.  The clients reserve their tokens before sending (llm._KVGate), so the count below must be zero; it is
# logged for every server this job started so that a run with preemptions is never used unnoticed.
preempt_report() {
  local f n
  for f in $P/logs/vllm_*_$P8.log $P/logs/vllm_*_$P4.log $P/logs/vllm_*_$P8B.log; do
    [ -e "$f" ] && [ "$f" -nt "$JSTART" ] || continue
    n=$(grep -c 'is preempted' "$f" 2>/dev/null || true)
    if [ "${n:-0}" != 0 ]; then log "WARNING: KV preemption in $(basename $f) ($n warning lines): outputs of this run may be corrupted"
    else log "no KV preemption in $(basename $f)"; fi
  done
  rm -f $JSTART
}
cleanup() { preempt_report || true; for pt in $P8 $P4 $P8B; do bash stop_vllm.sh $pt >/dev/null 2>&1 || true; done; }
trap cleanup EXIT
fin() { log "FAILED: $*"; exit 1; }
cnt() { find "$1" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l || true; }
export CM_P8=$P8 CM_PT=$P4 CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=$G1
# one 8B server alone on g1 ($1: optional minimum window; the job fails if the budget cannot provide it)
serve8() {
  local b=$(budget8 $FREE); local ml=$(maxlen8 $b)
  # a job that needs the full window takes exactly what that window needs when the card keeps 2.5 GB of headroom
  if [ "${1:-0}" = 24576 ] && [ "$ml" -lt 24576 ] && [ $((FREE-8600)) -ge 2500 ]; then b=8600; ml=24576; fi
  [ "$b" -ge "$MIN8" ] || fin "BUDGET_TOO_SMALL_8B budget=$b free=$FREE"
  if [ -n "${1:-}" ] && [ "$ml" -lt "$1" ]; then fin "WINDOW_TOO_SMALL budget=$b window=$ml need=$1"; fi
  export CM_MAXLEN=$ml
  J_FREE=$FREE bash start_srv.sh $G1 $P8 $Q8 qwen3-8b $(frac $b) $ml >> $P/logs/j_servers.log || fin SERVER_DOWN_8B
  log "8B server on GPU $G1 port $P8 budget=${b}MiB window=$ml"
}
# 8B compressor + int4 student ($1 student model name, or "none"); on one GPU or on g1/g2
serve_pair() {
  local b8 b4 ml
  if [ "$1" = none ]; then
    serve8 "${2:-}"; return
  fi
  if [ "$G2" = "$G1" ]; then
    b4=3950; [ "$FREE" -ge 20000 ] && b4=6000
    b8=$((FREE*75/100 - b4 - 300))
  else
    b8=$(budget8 $FREE); b4=$(budget4 $FREE2)
  fi
  ml=$(maxlen8 $b8)
  [ "$b8" -ge "$MIN8" ] || fin "BUDGET_TOO_SMALL_8B budget=$b8 free=$FREE"
  [ "$b4" -ge 3900 ] || fin "BUDGET_TOO_SMALL_4B budget=$b4 free2=$FREE2"
  if [ -n "${2:-}" ] && [ "$ml" -lt "$2" ]; then fin "WINDOW_TOO_SMALL budget=$b8 window=$ml need=$2"; fi
  export CM_MAXLEN=$ml
  J_FREE=$FREE bash start_srv.sh $G1 $P8 $Q8 qwen3-8b $(frac $b8) $ml >> $P/logs/j_servers.log || fin SERVER_DOWN_8B
  local md=$P/models/$1/merged; [ -e $NV/models/${1}_merged/config.json ] && md=$NV/models/${1}_merged
  bash start_tnet.sh $md $(frac $b4) $1 $P4 $G2 >> $P/logs/j_servers.log || fin "SERVER_DOWN_$1"
  up $P8 qwen3-8b || fin SERVER_DOWN_8B
  log "8B on GPU $G1 (budget ${b8}MiB, window $ml) + $1 on GPU $G2 (budget ${b4}MiB)"
}
# the student alone on g1, for a job whose 8B calls are all in the response cache (an uncached call finds no server
# and fails the job)
serve_student() {
  local b4=$(budget4 $FREE)
  [ "$b4" -ge 3900 ] || fin "BUDGET_TOO_SMALL_4B budget=$b4 free=$FREE"
  local md=$P/models/$1/merged; [ -e $NV/models/${1}_merged/config.json ] && md=$NV/models/${1}_merged
  bash start_tnet.sh $md $(frac $b4) $1 $P4 $G1 >> $P/logs/j_servers.log || fin "SERVER_DOWN_$1"
  log "$1 on GPU $G1 (budget ${b4}MiB); the 8B calls of this job are read from the response cache"
}
log "start gpus=$G port=$PORT free=$FREE/$FREE2"
