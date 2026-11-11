#!/bin/bash
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
export CUDA_VISIBLE_DEVICES=2 CM_EMB_DEVICE=cpu
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q5.log; }
log "A2 cfgen train:32"
$PY cfgen.py --splits train:32 --out $P/data/cf_train.jsonl --conc 160 >> $P/logs/q5_cfgen.log 2>&1
log "cfgen train rows $(wc -l < $P/data/cf_train.jsonl)"
touch $P/logs/Q5A.done
