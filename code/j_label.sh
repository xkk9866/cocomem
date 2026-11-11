#!/bin/bash
# v21 P1+P2: write-time derivation records (LME required users + LoCoMo), verified (delta, D, s) -> (D', s') targets,
# training sets of the five students (+ copies for the seed-1/2 students).
source j_common.sh
mkdir -p $P/states/deriv $P/states/deriv_locomo $P/models/d21
IFS=',' read -ra GS <<< "$G"
NS=$($PY deriv.py count --kind lme --required-only)
if [ "$(cnt $P/states/deriv)" -lt "$NS" ] || [ ! -e $P/models/d21/test.jsonl ]; then
  bash start_srv.sh ${GS[0]} $P8 $Q8 qwen3-8b $F8_ALONE >> $P/logs/j_servers.log || fin SERVER_DOWN
  REPL=$P8
  if [ ${#GS[@]} -ge 2 ]; then
    if bash start_srv.sh ${GS[1]} $P8B $Q8 qwen3-8b $F8_ALONE >> $P/logs/j_servers.log; then REPL=$P8,$P8B; else log "second replica failed; one replica"; fi
  fi
  export CM_Q8_REPLICAS=$REPL
  log "replicas $REPL"
  $PY deriv.py annotate --kind lme --required-only --conc 160 --inst 16 >> $P/logs/q40_deriv.log 2>&1 &
  d1=$!
  $PY deriv.py annotate --kind locomo --conc 48 >> $P/logs/q40_deriv_locomo.log 2>&1 &
  d2=$!
  wait $d1; wait $d2
  [ "$(cnt $P/states/deriv)" -ge "$NS" ] || fin "DERIV_INCOMPLETE $(cnt $P/states/deriv)/$NS"
  log "deriv lme=$(cnt $P/states/deriv)/$NS locomo=$(cnt $P/states/deriv_locomo)"
  [ -e $P/models/d21/test.jsonl ] || $PY commutator21.py relabel --src $P/models/w20 --out $P/models/d21 --conc 160 >> $P/logs/q40_label.log 2>&1
  unset CM_Q8_REPLICAS
fi
[ -e $P/models/d21/test.jsonl ] || fin RELABEL_FAILED
[ -e $P/models/tnet21nr/test.jsonl ] || $PY commutator21.py data --src $P/models/d21 >> $P/logs/q40_label.log 2>&1
for v in tnet21 tnet21nd tnet21w tnet21nv tnet21nr; do [ -s $P/models/$v/train.jsonl ] || fin "DATA_MISSING_$v"; done
for s in 1 2; do mkdir -p $P/models/tnet21_s$s; cp $P/models/tnet21/train.jsonl $P/models/tnet21/test.jsonl $P/models/tnet21_s$s/; done
log "label done: $(grep -E '^(train|test) ' $P/logs/q40_label.log | tail -2 | cut -c1-400 | tr '\n' ' ')"
touch $P/logs/J_label.done
