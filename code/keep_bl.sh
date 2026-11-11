#!/bin/bash
# keep_bl.sh : GPU 2 is shared with other users.  Keep the shared Qwen3-8B server (+ proxy) up on GPU 2: whenever it is down and
# the card has room (>= 20 GB free), (re)start it.  Never touches other GPUs.
P=/data/yaokang/corrmem; cd $P/code
while [ ! -e $P/logs/KEEP_BL.stop ]; do
  if ! curl -s -m 5 localhost:18440/v1/models | grep -q qwen3; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -d ' ')
    if [ "$used" -lt 4500 ]; then
      echo "[$(date '+%m-%d %H:%M:%S')] starting (used=$used)" >> $P/logs/keep_bl.log
      bash start_bl.sh 0.85 >> $P/logs/keep_bl.log 2>&1
    fi
  fi
  sleep 30
done
