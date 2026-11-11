#!/bin/bash
# Pause LLM clients, free GPU 2, train + evaluate NCF, restart vLLM, build the NCF index, resume clients.
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
CL=$(pgrep -f "run_corr.py|run_sys.py|locomo.py run|cfgen.py|judge.py" | tr '\n' ' ')
[ -n "$CL" ] && kill -STOP $CL
L=$(ss -ltnp 2>/dev/null | grep ":18431 " | grep -o 'pid=[0-9]*' | cut -d= -f2); [ -n "$L" ] && kill $L; sleep 20
CUDA_VISIBLE_DEVICES=2 $PY ncf.py train --data $P/data/cf_dev.jsonl --out $P/models/ncf > $P/logs/ncf_train.log 2>&1
CUDA_VISIBLE_DEVICES=2 $PY ncf.py eval --data $P/data/cf_test.jsonl --out $P/models/ncf --report $P/results/lineage_quality.json > $P/logs/ncf_eval.log 2>&1
CUDA_VISIBLE_DEVICES=2 CM_CLE_DEVICE=cuda $PY lineage_all.py > $P/logs/lineage_all.log 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.80
[ -n "$CL" ] && kill -CONT $CL
echo resumed $CL
