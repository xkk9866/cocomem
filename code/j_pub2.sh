#!/bin/bash
# Two-card run of the published baselines on GPUs 1 and 3 (0-3 only). Finished questions are skipped.
set -Eeuo pipefail
P=/data/yaokang/corrmem
cd $P/code
PY=../.venv/bin/python
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CM_EMB_DEVICE=cpu
bash start_srv.sh 1 18620 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.55 16384
SECOND=0
bash start_srv.sh 3 18621 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.40 8192 && SECOND=1 || echo "GPU3 server unavailable; one card"
CM_P8=18620 $PY run_sys.py --systems ga,readagent --splits test200 --tag main3 --inst 4 --conc 24 --shard 0:2 \
  > $P/logs/pub_s0.log 2>&1 &
if [ "$SECOND" = 1 ]; then
  CM_P8=18621 $PY run_sys.py --systems ga,readagent --splits test200 --tag main3 --inst 4 --conc 24 --shard 1:2 \
    > $P/logs/pub_s1.log 2>&1 &
else
  CM_P8=18620 $PY run_sys.py --systems ga,readagent --splits test200 --tag main3 --inst 4 --conc 24 --shard 1:2 \
    > $P/logs/pub_s1.log 2>&1 &
fi
wait
echo GA_RA_DONE
CM_P8=18620 $PY locomo.py run --methods ga,readagent > $P/logs/pub_loc.log 2>&1
# the second server is idle during the ten LoCoMo conversations; leave it up for A-Mem
# one judge, then A-Mem on both cards
bash start_srv.sh 1 18620 /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.55 12288
$PY judge.py --model qwen25-7b --port 18620 --glob \
  "$P/results/main3/ga/*.json,$P/results/main3/readagent/*.json,$P/results/locomo/ga/*.json,$P/results/locomo/readagent/*.json"
bash start_srv.sh 1 18620 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.55 16384
SECOND=0
bash start_srv.sh 3 18621 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.40 8192 && SECOND=1 || echo "GPU3 server unavailable for A-Mem"
CM_P8=18620 $PY run_sys.py --systems amem --splits test200 --tag main3 --inst 2 --conc 12 --shard 0:2 > $P/logs/pub_am0.log 2>&1 &
if [ "$SECOND" = 1 ]; then
  CM_P8=18621 $PY run_sys.py --systems amem --splits test200 --tag main3 --inst 2 --conc 12 --shard 1:2 > $P/logs/pub_am1.log 2>&1 &
else
  CM_P8=18620 $PY run_sys.py --systems amem --splits test200 --tag main3 --inst 2 --conc 12 --shard 1:2 > $P/logs/pub_am1.log 2>&1 &
fi
wait
CM_P8=18620 $PY locomo.py run --methods amem
bash start_srv.sh 1 18620 /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.55 12288
$PY judge.py --model qwen25-7b --port 18620 --glob "$P/results/main3/amem/*.json,$P/results/locomo/amem/*.json"
echo PUB_DONE
touch $P/logs/PUB.done
