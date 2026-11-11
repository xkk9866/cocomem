#!/bin/bash
# v4 stage A (GPU 2): train-split memories -> compressor counterfactual re-derivations (commutator distillation data)
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
export CUDA_VISIBLE_DEVICES=2 CM_EMB_DEVICE=cpu
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $P/logs/q5.log; }
log "A start"
$PY run_build.py --splits train --inst 10 --conc 128 >> $P/logs/q5_build.log 2>&1
log "train built $(ls $P/states/cocomem | wc -l)"
$PY cfgen.py --splits train --out $P/data/cf_train.jsonl --conc 160 >> $P/logs/q5_cfgen.log 2>&1
log "cfgen train rows $(wc -l < $P/data/cf_train.jsonl)"
touch $P/logs/Q5A.done
