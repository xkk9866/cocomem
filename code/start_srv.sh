#!/bin/bash
# start_srv.sh GPU PORT MODEL_DIR NAME FRAC [MAXLEN] : int4 (AWQ) vLLM server on one shared GPU.
# FRAC is the share of the card the server may use; the batch configuration adapts to that budget so that the
# footprint stays inside it (eager mode, chunked prefill, bounded sequences).  Exits as soon as the server dies.
P=/data/yaokang/corrmem; cd $P/code
G=$1; PT=$2; M=$3; N=$4; F=${5:-0.5}; ML=${6:-24576}
if curl -sf localhost:$PT/v1/models | grep -q "\"$N\""; then echo up; exit 0; fi
L=$(ss -ltnp 2>/dev/null | grep ":$PT " | grep -o 'pid=[0-9]*' | cut -d= -f2); [ -n "$L" ] && kill $L 2>/dev/null && sleep 10
B=$(python3 -c "print(int($F*24564))")
if [ "$B" -ge 20000 ]; then
  batch=(--max-num-seqs 128)
elif [ "$B" -ge 12000 ]; then
  batch=(--max-num-seqs 64 --enable-chunked-prefill --max-num-batched-tokens 4096 --enforce-eager)
elif [ "$B" -ge 8600 ]; then
  batch=(--max-num-seqs 32 --enable-chunked-prefill --max-num-batched-tokens 2048 --enforce-eager)
else
  batch=(--max-num-seqs 16 --enable-chunked-prefill --max-num-batched-tokens 1024 --enforce-eager)
fi
CUDA_VISIBLE_DEVICES=$G VLLM_USE_V1=0 setsid nohup ../.venv/bin/python -m vllm.entrypoints.openai.api_server --model $M \
  --served-model-name $N --port $PT --max-model-len $ML --gpu-memory-utilization $F \
  "${batch[@]}" --kv-cache-dtype fp8 --disable-log-requests --enable-prefix-caching > $P/logs/vllm_${N}_$PT.log 2>&1 < /dev/null 9>&- &
pid=$!
for i in $(seq 1 240); do
  sleep 5
  curl -s localhost:$PT/v1/models | grep -q "\"$N\"" && { echo "up budget=${B}MiB window=$ML"; exit 0; }
  if ! kill -0 $pid 2>/dev/null; then echo "server exited: $N port=$PT gpu=$G frac=$F budget=$B window=$ML"; grep -m2 -E 'OutOfMemoryError|ValueError|Error' $P/logs/vllm_${N}_$PT.log | cut -c1-240; exit 1; fi
done
echo "failed: $N port=$PT gpu=$G did not come up in 20 min (killing pid $pid)"; kill $pid 2>/dev/null; sleep 5; kill -9 $pid 2>/dev/null; exit 1
