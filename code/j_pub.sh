#!/bin/bash
# Published memory systems on the same correction sets as CoCoMem. GPUs 0-3 only.
set -Eeuo pipefail
P=/data/yaokang/corrmem
cd $P/code
PY=../.venv/bin/python
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CM_EMB_DEVICE=cpu
# freest card among 0-3
read -r G FREE < <($PY - <<'PY'
import subprocess
rows = []
for line in subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.total","--format=csv,noheader,nounits"], text=True).splitlines():
    i, u, t = [int(x.strip()) for x in line.split(",")]
    if i <= 3:
        rows.append((t - u, i))
rows.sort(reverse=True)
print(rows[0][1], rows[0][0])
PY
)
echo "gpu=$G free=$FREE"
F=$(python3 -c "print(max(0.20, min(0.70, round(($FREE-800)/24564, 3))))")
export J_FREE=$FREE
bash start_srv.sh $G 18620 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b $F 16384
export CM_P8=18620
# Generative Agents and ReadAgent: one model call per session, then the shared reader
$PY run_sys.py --systems ga,readagent --splits test200 --tag main3 --inst 4 --conc 32
$PY locomo.py run --methods ga,readagent
bash start_srv.sh $G 18620 /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b $F 12288
$PY judge.py --model qwen25-7b --port 18620 --glob "$P/results/main3/ga/*.json,$P/results/main3/readagent/*.json,$P/results/locomo/ga/*.json,$P/results/locomo/readagent/*.json"
# A-Mem uses the authors' analysis and evolution prompts, one note per dialogue round
bash start_srv.sh $G 18620 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b $F 16384
$PY run_sys.py --systems amem --splits test200 --tag main3 --inst 2 --conc 16
$PY locomo.py run --methods amem
bash start_srv.sh $G 18620 /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b $F 12288
$PY judge.py --model qwen25-7b --port 18620 --glob "$P/results/main3/amem/*.json,$P/results/locomo/amem/*.json"
echo PUB_DONE
touch $P/logs/PUB.done
