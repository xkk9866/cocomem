#!/bin/bash
# v22 P3b (CPU): merge the relabel shards, build the training sets of tnet22 and its ablations (+ seed copies).
source j_common.sh
export CM_RUN_ROOT=$P/v22 CM_COMPRESSOR=plan
trap - EXIT
N=$1
OUT=$P/models/d22
[ -e $OUT/test.jsonl ] || $PY commutator21.py merge --out $OUT --shards $N >> $P/logs/v22_label.log 2>&1 || fin MERGE_FAILED
[ -e $P/models/tnet22nr/test.jsonl ] || $PY commutator21.py data --src $OUT --prefix tnet22 --keyed >> $P/logs/v22_label.log 2>&1 || fin DATA_FAILED
for v in tnet22 tnet22nd tnet22w tnet22nv tnet22nr; do [ -s $P/models/$v/train.jsonl ] || fin "DATA_MISSING_$v"; done
for s in 1 2; do mkdir -p $P/models/tnet22_s$s; cp $P/models/tnet22/train.jsonl $P/models/tnet22/test.jsonl $P/models/tnet22_s$s/; done
mkdir -p $CM_RUN_ROOT/results
cp $OUT/label_cost.json $CM_RUN_ROOT/results/label_cost_v22.json 2>/dev/null || true
log "label merge done: $(grep -E '^(train|test) [0-9]' $P/logs/v22_label.log | tail -2 | cut -c1-200 | tr '\n' ' ')"
touch $P/logs/J_$J_JOB.done
