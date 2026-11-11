#!/bin/bash
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
export CUDA_VISIBLE_DEVICES=2
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.70 > $P/logs/q11.log
bash start_tnet.sh $P/models/tnet2/merged 0.20 tnet2 18432 >> $P/logs/q11.log
bash start_tnet.sh $P/models/tnet3/merged 0.20 tnet3 18433 1 >> $P/logs/q11.log
$PY suffbench.py --units 30 --chain 2 > $P/logs/q11_suff.log 2>&1
cp $P/results/suffbench_cases.json $P/results/suffbench_cases_probe.json
touch $P/logs/Q11.done
