#!/bin/bash
cd /data/yaokang/corrmem/code
export CUDA_VISIBLE_DEVICES=2 CM_EMB_DEVICE=cpu
PY=../.venv/bin/python
$PY judge.py --model qwen3-8b --glob "/data/yaokang/corrmem/results/main3/ccf/*.json,/data/yaokang/corrmem/results/main3/slots/*.json,/data/yaokang/corrmem/results/main3/append/*.json,/data/yaokang/corrmem/results/main3/recompress/*.json,/data/yaokang/corrmem/results/main3/coarse/*.json,/data/yaokang/corrmem/results/main3/self_span/*.json,/data/yaokang/corrmem/results/main3/simupdate/*.json,/data/yaokang/corrmem/results/main3/ccf_overwrite/*.json" > ../logs/prev_judge.log 2>&1
$PY analyze.py --tag main3 --methods append,slots,simupdate,self_span,coarse,recompress,ccf,ccf_overwrite,rawrag,membank,mem0 --ref ccf --key label_qwen3-8b > ../results/prev_main3.txt 2>&1
$PY comm_eval.py --tag main3 --model qwen3-8b --methods append,slots,simupdate,self_span,coarse,ccf,recompress > ../logs/prev_comm.log 2>&1
touch ../logs/prev.done
