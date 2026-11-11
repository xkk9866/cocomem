#!/bin/bash
# Run the reader alongside one student on GPU 6.
set -eu
P=/data/yaokang/corrmem
cd "$P/code"
model=/home/yaokang/models/Qwen3-8B-AWQ
port=18431
ready() {
  curl -fsS --max-time 3 "localhost:$port/v1/models" 2>/dev/null |
    "$P/.venv/bin/python" -c 'import json,sys; sys.exit(not any(x["id"]=="qwen3-8b" and x["root"]==sys.argv[1] for x in json.load(sys.stdin)["data"]))' "$model" 2>/dev/null
}
if ready; then exit 0; fi
CUDA_VISIBLE_DEVICES=6 VLLM_USE_V1=0 setsid nohup "$P/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$model" --served-model-name qwen3-8b --port "$port" --max-model-len 24576 \
  --gpu-memory-utilization 0.45 --max-num-seqs 8 --enable-chunked-prefill --max-num-batched-tokens 1024 \
  --kv-cache-dtype fp8 --enforce-eager --enable-prefix-caching --disable-log-requests \
  > "$P/logs/q31_gpu6_reader.log" 2>&1 < /dev/null 9>&- &
pid=$!
for i in $(seq 1 300); do
  if ready; then exit 0; fi
  kill -0 "$pid" 2>/dev/null || exit 1
  sleep 4
done
exit 1
