#!/bin/bash
set -Eeuo pipefail
P=/data/yaokang/corrmem
cd "$P/code"
exec 8>"$P/logs/q31.wait.lock"
flock -n 8 || exit 1
exec 9>"$P/logs/q30.run.lock"
flock -n 9 || exit 1
bash stop_vllm.sh 18431
printf '%s waiting for GPUs 6,7; checkpoints retained\n' "$(date -Is)" >> "$P/logs/q31_wait.log"
ready=0
while [ "$ready" -lt 3 ]; do
  if nvidia-smi -i 6,7 --query-gpu=memory.free --format=csv,noheader,nounits | awk 'BEGIN {ok=1; n=0} {n++; if ($1<22000) ok=0} END {exit !(ok && n==2)}'; then
    ready=$((ready+1))
  else
    ready=0
  fi
  sleep 10
done
printf '%s GPUs available; resuming q31\n' "$(date -Is)" >> "$P/logs/q31_wait.log"
flock -u 9
exec 9>&-
flock -u 8
exec 8>&-
exec bash q31_multiseed.sh
