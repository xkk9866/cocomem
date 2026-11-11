#!/bin/bash
# Stage A (GPU 2): counterfactual self-supervision + oracle validity runs, in parallel on one vLLM server.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python; export CM_EMB_DEVICE=cpu
$PY cfgen.py --splits dev --out $P/data/cf_dev.jsonl > $P/logs/cf_dev.log 2>&1 &
$PY cfgen.py --splits test200:10 --out $P/data/cf_test.jsonl --seed 1 > $P/logs/cf_test.log 2>&1 &
$PY run_sys.py --systems oracle --splits dev,test200 --tag main --inst 8 > $P/logs/oracle2.log 2>&1 &
wait
echo done > $P/logs/A.done
