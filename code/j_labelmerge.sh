#!/bin/bash
# v21 P2b (CPU): merge the relabel shards, build the training sets of the five students (+ seed copies).
source j_common.sh
trap - EXIT
N=$1
[ -e $P/models/d21/test.jsonl ] || $PY commutator21.py merge --out $P/models/d21 --shards $N >> $P/logs/q40_label.log 2>&1 || fin MERGE_FAILED
[ -e $P/models/tnet21nr/test.jsonl ] || $PY commutator21.py data --src $P/models/d21 >> $P/logs/q40_label.log 2>&1 || fin DATA_FAILED
for v in tnet21 tnet21nd tnet21w tnet21nv tnet21nr; do [ -s $P/models/$v/train.jsonl ] || fin "DATA_MISSING_$v"; done
for s in 1 2; do mkdir -p $P/models/tnet21_s$s; cp $P/models/tnet21/train.jsonl $P/models/tnet21/test.jsonl $P/models/tnet21_s$s/; done
cp $P/models/d21/label_cost.json $P/results/label_cost_v21.json 2>/dev/null || true
log "label merge done: $(grep -E '^(train|test) [0-9]' $P/logs/q40_label.log | tail -2 | cut -c1-200 | tr '\n' ' ')"
touch $P/logs/J_$J_JOB.done
