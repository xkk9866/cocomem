#!/bin/bash
# Serve or rejoin an exact student model on the requested GPU and port.
P=/data/yaokang/corrmem; cd "$P/code"
M=${1:-$P/models/tnet/merged}; F=${2:-0.20}; N=${3:-tnet}; PT=${4:-18432}
if curl -sf localhost:$PT/v1/models | grep -q "\"$N\""; then echo up; exit 0; fi
# Startup may still be loading weights after a previous queue timeout.
server_pid=$(../.venv/bin/python - "$M" "$PT" <<'PY'
import os, sys
from pathlib import Path
for d in Path('/proc').iterdir():
    if not d.name.isdigit(): continue
    try:
        a=(d/'cmdline').read_bytes().decode().split('\0')
        if d.stat().st_uid != os.getuid(): continue
        if 'vllm.entrypoints.openai.api_server' not in a: continue
        if a[a.index('--model')+1] == sys.argv[1] and a[a.index('--port')+1] == sys.argv[2]:
            print(d.name); break
    except (OSError, ValueError, IndexError, UnicodeError): pass
PY
)
if [ -z "$server_pid" ]; then
  bash stop_vllm.sh "$PT"
  # the operator's prompts are short (correction + record + sentence; pointer witnesses up to a few thousand
  # tokens): an 8k window with an fp8 KV cache keeps the student inside a few GB of a shared card
  B=$(python3 -c "print(int($F*24564))"); NS=32; [ "$B" -ge 9000 ] && NS=128
  CUDA_VISIBLE_DEVICES=${5:-2} VLLM_USE_V1=0 setsid nohup ../.venv/bin/python -m vllm.entrypoints.openai.api_server --model "$M" \
    --served-model-name "$N" --port "$PT" --max-model-len 8192 --gpu-memory-utilization "$F" --max-num-seqs $NS --enforce-eager \
    --enable-chunked-prefill --max-num-batched-tokens 2048 --kv-cache-dtype fp8 \
    --quantization bitsandbytes --load-format bitsandbytes --disable-log-requests --enable-prefix-caching \
    > "$P/logs/vllm_tnet_$PT.log" 2>&1 < /dev/null 9>&- &
  server_pid=$!
fi
for i in $(seq 1 360); do
  if curl -sf localhost:$PT/v1/models | grep -q "\"$N\""; then echo up; exit 0; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then echo "student server exited: $N pid=$server_pid"; tail -25 "$P/logs/vllm_tnet_$PT.log"; exit 1; fi
  sleep 5
done
echo "student startup timeout: $N pid=$server_pid (killed)"; tail -20 "$P/logs/vllm_tnet_$PT.log"; kill "$server_pid" 2>/dev/null; sleep 5; kill -9 "$server_pid" 2>/dev/null; exit 1
