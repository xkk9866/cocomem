#!/bin/bash
# v22 P3: commutation-verified (delta, D, s) -> (D', s') targets on the record-first memories (commutator21.py relabel
# with CM_RUN_ROOT=v22: the units' own records, no second pass), 8B compressor alone.
#   j22_relabel.sh I N     shard I of N of the pairs on a reduced window; pairs whose teacher calls need the full
#                          window (profile units) are kept as deferred copies
#   j22_relabel.sh redo N  second pass on the full window: only the deferred pairs
source j_common.sh
export CM_RUN_ROOT=$P/v22 CM_COMPRESSOR=plan
MODE=$1; N=$2
SRC=$P/models/w20
OUT=$P/models/d22
mkdir -p $OUT
count_deferred() { $PY - <<EOF
import json, glob
n = 0
for f in glob.glob("$OUT/*.s*of$N.jsonl"):
    n += sum(bool(json.loads(l).get("deferred")) for l in open(f))
print(n)
EOF
}
if [ "$MODE" = redo ]; then
  for I in $(seq 0 $((N-1))); do [ -e $OUT/test.s${I}of$N.jsonl ] || fin "SHARD_MISSING_$I"; done
  LEFT=$(count_deferred)
  if [ "$LEFT" != 0 ]; then
    serve8 24576
    for I in $(seq 0 $((N-1))); do
      $PY commutator21.py relabel --src $SRC --out $OUT --shard $I:$N --redo-deferred --propose-more --conc 48 >> $P/logs/v22_label_redo.log 2>&1 || fin RELABEL_REDO_FAILED
    done
    LEFT=$(count_deferred)
  fi
  log "relabel redo done: deferred pairs left=$LEFT"
  [ "$LEFT" = 0 ] || fin "DEFERRED_LEFT_$LEFT"
else
  I=$MODE
  # a pair is labelled on the sentences of its unit as written record-first: every pair unit must be written by now
  read -r NM ND NP NU NPP < <($PY rebuild_units.py count --kind lme)
  [ "$NPP" = 0 ] || fin "PAIR_UNITS_PENDING_$NPP"
  if [ ! -e $OUT/test.s${I}of$N.jsonl ] && [ ! -e $OUT/test.jsonl ]; then
    serve8
    CONC=48; [ "$FREE" -ge 12000 ] && CONC=128
    $PY commutator21.py relabel --src $SRC --out $OUT --shard $I:$N --propose-more --conc $CONC >> $P/logs/v22_label_$I.log 2>&1 || fin RELABEL_FAILED
  fi
  [ -e $OUT/test.s${I}of$N.jsonl ] || [ -e $OUT/test.jsonl ] || fin RELABEL_OUTPUT_MISSING
  log "relabel shard $I/$N done: $(grep -E '^(train|test) [0-9]' $P/logs/v22_label_$I.log | tail -2 | cut -c1-300 | tr '\n' ' ')"
fi
touch $P/logs/J_$J_JOB.done
