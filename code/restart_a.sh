#!/bin/bash
# (Re)start counterfactual generation + the stage-B pipeline on GPU 2.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python; export CM_EMB_DEVICE=cpu
curl -s localhost:18431/v1/models | grep -q qwen3-8b || bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.72
rm -f $P/data/cf_dev.jsonl $P/data/cf_test.jsonl $P/logs/stageB.log
setsid nohup $PY cfgen.py --splits dev --out $P/data/cf_dev.jsonl > $P/logs/cf_dev.log 2>&1 < /dev/null &
setsid nohup $PY cfgen.py --splits test200:10 --out $P/data/cf_test.jsonl --seed 1 > $P/logs/cf_test.log 2>&1 < /dev/null &
pgrep -f "run_sys.py --systems oracle" > /dev/null || setsid nohup $PY run_sys.py --systems oracle --splits dev,test200 --tag main --inst 8 >> $P/logs/oracle2.log 2>&1 < /dev/null &
sleep 30
setsid nohup bash jobs_b.sh > /dev/null 2>&1 < /dev/null &
echo restarted
