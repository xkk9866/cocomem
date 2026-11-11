#!/bin/bash
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
export CM_EMB_DEVICE=cpu CUDA_VISIBLE_DEVICES=2
log() { echo "[$(date +%H:%M:%S)] $*" >> $P/logs/judge_final.log; }
bash start_gpu2.sh /data/yaokang/evishare/models/Qwen2.5-7B-Instruct-AWQ qwen25-7b 0.74 >> $P/logs/post_q3.log
$PY judge.py --model qwen25-7b --glob "$P/results/main/*/*.json,$P/results/locomo/*/*.json" >> $P/logs/judge_q25.log 2>&1
M=append,simupdate,self,self_span,coarse,ncf,recompress,spec_ncf,spec_overwrite,rebuild,spec,rawrag,membank,mem0,oracle
$PY analyze.py --tag main --methods $M --ref ncf --key label_qwen25-7b --json $P/results/table_main.json > $P/results/table_main.txt 2>&1
$PY analyze.py --tag main --methods $M --ref ncf --key label_qwen3-8b --json $P/results/table_main_q3.json > $P/results/table_main_q3judge.txt 2>&1
$PY analyze.py --tag main --methods $M --ref ncf --key label_qwen25-7b --all > $P/results/table_main_all.txt 2>&1
$PY analyze.py --tag locomo --methods ncf,spec,append,simupdate,self,coarse,recompress,rebuild,rawrag,membank,mem0,amem --key label_qwen25-7b --json $P/results/table_locomo.json > $P/results/table_locomo.txt 2>&1
$PY gen_tables.py $P/results $P/results/tables.tex > /dev/null 2>&1
bash start_gpu2.sh /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b 0.74 >> $P/logs/post_q3.log
log "POST DONE"; touch $P/logs/POST.done
