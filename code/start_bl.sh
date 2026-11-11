#!/bin/bash
# start_bl.sh : shared Qwen3-8B-AWQ server (GPU 2 only) with tool calling (LangMem) + the proxy used by the official baselines
P=/data/yaokang/corrmem; cd $P/code
bash stop_vllm.sh 18433 >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=2 VLLM_USE_V1=0 setsid nohup ../.venv/bin/python -m vllm.entrypoints.openai.api_server --model /home/yaokang/models/Qwen3-8B-AWQ \
  --served-model-name qwen3-8b --port 18433 --max-model-len 24576 --gpu-memory-utilization ${1:-0.85} \
  --max-num-seqs 128 --kv-cache-dtype fp8 --disable-log-requests --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes > $P/logs/vllm_bl_18433.log 2>&1 < /dev/null &
for i in $(seq 1 240); do sleep 5; curl -s localhost:18433/v1/models | grep -q '"qwen3-8b"' && break; done
curl -s localhost:18433/v1/models | grep -q '"qwen3-8b"' || { echo failed; exit 1; }
pkill -f "llm_proxy.py 18440" ; sleep 1
setsid nohup ../.venv/bin/python llm_proxy.py 18440 18433 > $P/logs/proxy.log 2>&1 < /dev/null &
sleep 3; curl -s localhost:18440/v1/models | head -c 100; echo; echo up
