#!/bin/bash
# restart vLLM (8B) and relaunch background jobs listed in jobs.txt
cd /data/yaokang/corrmem
for p in $(pgrep -f "python (gen_corr|run_build|run_sys|run_corr|cla|locomo|judge).py"); do kill $p; done
[ -f logs/vllm_qwen3-8b.pid ] && kill $(cat logs/vllm_qwen3-8b.pid) 2>/dev/null
sleep 5
bash serve.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 18431 20480 0.82
for i in $(seq 1 60); do sleep 5; curl -s localhost:18431/v1/models >/dev/null && break; done
curl -s localhost:18431/v1/models | head -c 40; echo
cd code
while read -r line; do
  [ -z "$line" ] && continue
  name=$(echo "$line" | cut -d'|' -f1); cmd=$(echo "$line" | cut -d'|' -f2-)
  CM_EMB_DEVICE=cpu nohup bash -c "$cmd" > ../logs/$name.log 2>&1 &
done < ../jobs.txt
sleep 2; ps -eo pid,cmd | grep -E "python (gen_corr|run_build|run_sys|run_corr)" | grep -v grep
