#!/bin/bash
# Start the vLLM server on GPU 2 only.  $1 model dir (default Qwen3-8B-AWQ)  $2 served name  $3 mem fraction
P=/data/yaokang/corrmem
cd $P/code
M=${1:-/home/yaokang/models/Qwen3-8B-AWQ}; N=${2:-qwen3-8b}; F=${3:-0.72}
[ -f $P/logs/vllm_cur.pid ] && kill $(cat $P/logs/vllm_cur.pid) 2>/dev/null
L=$(ss -ltnp 2>/dev/null | grep ":18431 " | grep -o 'pid=[0-9]*' | cut -d= -f2)
[ -n "$L" ] && kill $L 2>/dev/null
sleep 12
CUDA_VISIBLE_DEVICES=2 VLLM_USE_V1=0 setsid nohup ../.venv/bin/python -m vllm.entrypoints.openai.api_server --model $M \
  --served-model-name $N --port 18431 --max-model-len 24576 --gpu-memory-utilization $F \
  --max-num-seqs 128 --kv-cache-dtype fp8 --disable-log-requests --enable-prefix-caching > $P/logs/vllm_$N.log 2>&1 < /dev/null &
echo $! > $P/logs/vllm_cur.pid
for i in $(seq 1 240); do sleep 5; curl -s localhost:18431/v1/models | grep -q "\"$N\"" && { echo up; exit 0; }; done
echo failed; tail -20 $P/logs/vllm_$N.log; exit 1
