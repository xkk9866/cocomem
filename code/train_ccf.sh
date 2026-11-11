#!/bin/bash
# Train the correction-conditioned commutator field on GPU 2 (vLLM paused), evaluate, restart vLLM.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
L=$(ss -ltnp 2>/dev/null | grep ":18431 " | grep -o 'pid=[0-9]*' | cut -d= -f2); [ -n "$L" ] && kill $L; sleep 20
CUDA_VISIBLE_DEVICES=2 $PY ncf.py train --data $P/data/cf_dev.jsonl --out $P/models/ccf --epochs 6 > $P/logs/ccf_train.log 2>&1
CUDA_VISIBLE_DEVICES=2 $PY ncf.py eval --data $P/data/cf_test.jsonl --out $P/models/ccf --report $P/results/lineage_quality_ccf.json > $P/logs/ccf_eval.log 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.74 >> $P/logs/ccf_train.log
touch $P/logs/CCF.done
