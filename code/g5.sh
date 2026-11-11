#!/bin/bash
# GPU 5 side lane: Llama-3.1-8B (int4) judge server on port 18441; direct commutativity test of the existing strategies.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
CUDA_VISIBLE_DEVICES=5 VLLM_USE_V1=0 setsid nohup ../.venv/bin/python -m vllm.entrypoints.openai.api_server   --model /data/yaokang/models/Llama-3.1-8B-Instruct-AWQ --served-model-name llama31 --port 18441 --max-model-len 8192   --gpu-memory-utilization 0.30 --max-num-seqs 64 --kv-cache-dtype fp8 --disable-log-requests --enable-prefix-caching   > $P/logs/vllm_llama_g5.log 2>&1 < /dev/null &
for i in $(seq 1 120); do sleep 5; curl -s localhost:18441/v1/models | grep -q llama31 && break; done
curl -s localhost:18441/v1/models | grep -q llama31 || { echo "[$(date)] g5 server failed" >> $P/logs/q5.log; exit 1; }
echo "[$(date '+%m-%d %H:%M:%S')] g5 llama up" >> $P/logs/q5.log
$PY comm_eval2.py --model llama31 --port 18441 --suffix _old   --methods append,slots,simupdate,self_span,coarse,ncf,ccf,spec,recompress,rebuild > $P/logs/g5_comm2.log 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] g5 comm2 old done" >> $P/logs/q5.log
