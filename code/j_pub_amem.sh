#!/bin/bash
# A-Mem on the free cards among 0-3. Two shards when two cards have room.
set -Eeuo pipefail
P=/data/yaokang/corrmem
cd $P/code
PY=../.venv/bin/python
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CM_EMB_DEVICE=cpu
rm -f $P/logs/AMEM.done
pick() {
  $PY - <<'PY'
import subprocess
rows = []
for line in subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.total","--format=csv,noheader,nounits"], text=True).splitlines():
    i, u, t = [int(x.strip()) for x in line.split(",")]
    if i <= 3:
        rows.append((t - u, i, t))
rows.sort(reverse=True)
for free, i, t in rows:
    if free >= 12000:
        print(i, free, t)
PY
}
mapfile -t CAND < <(pick)
echo "candidates ${#CAND[@]}"
[ "${#CAND[@]}" -ge 1 ]
read -r G1 F1 T1 <<<"${CAND[0]}"
FRAC() { python3 -c "print(round(max(0.20, min(0.40, ($1-2500)/$2)), 3))"; }
F=$(FRAC "$F1" "$T1")
bash start_srv.sh "$G1" 18622 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b "$F" 8192
if [ "${#CAND[@]}" -ge 2 ]; then
  read -r G2 F2 T2 <<<"${CAND[1]}"
  F=$(FRAC "$F2" "$T2")
  bash start_srv.sh "$G2" 18623 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b "$F" 8192
  CM_P8=18622 $PY run_sys.py --systems amem --splits test200 --tag main3 --inst 6 --conc 12 --shard 0:2 > $P/logs/pub_am0.log 2>&1 &
  CM_P8=18623 $PY run_sys.py --systems amem --splits test200 --tag main3 --inst 6 --conc 12 --shard 1:2 > $P/logs/pub_am1.log 2>&1 &
  wait
  bash stop_vllm.sh 18623 || true
else
  CM_P8=18622 $PY run_sys.py --systems amem --splits test200 --tag main3 --inst 6 --conc 12
fi
rm -f $P/results/locomo/amem/*.json
CM_P8=18622 $PY locomo.py run --methods amem
bash stop_vllm.sh 18622 || true
echo AMEM_GEN_DONE
touch $P/logs/AMEM.done
