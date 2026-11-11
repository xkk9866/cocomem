#!/bin/bash
# Finish the published baselines. Finished question files are skipped.
set -Eeuo pipefail
P=/data/yaokang/corrmem
cd $P/code
PY=../.venv/bin/python
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CM_EMB_DEVICE=cpu
pick() {
  $PY - <<'PY'
import subprocess
best = None
for line in subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.total","--format=csv,noheader,nounits"], text=True).splitlines():
    i, u, t = [int(x.strip()) for x in line.split(",")]
    if i <= 3:
        free = t - u
        if best is None or free > best[0]:
            best = (free, i, t)
print(best[1], best[0], best[2])
PY
}
serve() {
  local model=$1 name=$2 ml=$3 tries=0
  while [ "$tries" -lt 8 ]; do
    tries=$((tries+1))
    read -r G FREE TOTAL < <(pick)
    echo "try $tries gpu=$G free=$FREE model=$name"
    if [ "$FREE" -lt 9000 ]; then sleep 30; continue; fi
    F=$(python3 -c "print(round(max(0.20, min(0.40, ($FREE-2500)/$TOTAL)), 3))")
    if bash start_srv.sh $G 18620 "$model" "$name" "$F" "$ml"; then
      export J_FREE=$FREE
      return 0
    fi
    bash stop_vllm.sh 18620 || true
    sleep 20
  done
  echo "no card for $name"; exit 1
}
export CM_P8=18620
serve /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 8192
$PY run_sys.py --systems ga,readagent --splits test200 --tag main3 --inst 4 --conc 24 --shard 1:2
$PY locomo.py run --methods ga,readagent
bash stop_vllm.sh 18620 || true
serve /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 8192
$PY judge.py --model qwen25-7b --port 18620 --glob \
  "$P/results/main3/ga/*.json,$P/results/main3/readagent/*.json,$P/results/locomo/ga/*.json,$P/results/locomo/readagent/*.json"
bash stop_vllm.sh 18620 || true
while [ ! -f $P/logs/AMEM.done ]; do echo "waiting for amem"; sleep 60; done
serve /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 8192
$PY judge.py --model qwen25-7b --port 18620 --glob "$P/results/main3/amem/*.json,$P/results/locomo/amem/*.json"
echo PUB_DONE
touch $P/logs/PUB.done
