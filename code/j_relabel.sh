#!/bin/bash
# v21 P2: commutation-verified (delta, D, s) -> (D', s') targets (commutator21.py relabel), 8B compressor alone.
#   j_relabel.sh I N     shard I of N of the pairs on a reduced window; pairs whose teacher calls need the full
#                        window (profile units: all claims are listed to the verifier) are kept as deferred copies
#   j_relabel.sh redo N  second pass on the full window: only the deferred pairs of every shard are re-labelled
source j_common.sh
MODE=$1; N=$2
PAIRS=$P/models/w20/train.jsonl,$P/models/w20/test.jsonl
mkdir -p $P/models/d21
if [ "$MODE" = redo ]; then
  read -r PF PD PU PH < <($PY deriv.py count --kind lme --required-only --units-from $PAIRS)
  [ "$PH" -ge "$PU" ] || fin "DERIV_INCOMPLETE pair units $PH/$PU"
  for I in $(seq 0 $((N-1))); do [ -e $P/models/d21/test.s${I}of$N.jsonl ] || fin "SHARD_MISSING_$I"; done
  count_deferred() { $PY - <<EOF
import json, glob
n = 0
for f in glob.glob("$P/models/d21/*.s*of$N.jsonl"):
    n += sum(bool(json.loads(l).get("deferred")) for l in open(f))
print(n)
EOF
  }
  LEFT=$(count_deferred)
  if [ "$LEFT" != 0 ]; then
    serve8 24576
    for I in $(seq 0 $((N-1))); do
      $PY commutator21.py relabel --src $P/models/w20 --out $P/models/d21 --shard $I:$N --redo-deferred --conc 48 >> $P/logs/q40_label_redo.log 2>&1 || fin RELABEL_REDO_FAILED
    done
    LEFT=$(count_deferred)
  fi
  log "relabel redo done: deferred pairs left=$LEFT"
  [ "$LEFT" = 0 ] || fin "DEFERRED_LEFT_$LEFT"
else
  I=$MODE
  if [ ! -e $P/models/d21/test.s${I}of$N.jsonl ] && [ ! -e $P/models/d21/test.jsonl ]; then
    serve8
    CONC=48; [ "$FREE" -ge 12000 ] && CONC=128
    $PY commutator21.py relabel --src $P/models/w20 --out $P/models/d21 --shard $I:$N --conc $CONC >> $P/logs/q40_label_$I.log 2>&1 || fin RELABEL_FAILED
  fi
  [ -e $P/models/d21/test.s${I}of$N.jsonl ] || [ -e $P/models/d21/test.jsonl ] || fin RELABEL_OUTPUT_MISSING
  log "relabel shard $I/$N done: $(grep -E '^(train|test) [0-9]' $P/logs/q40_label_$I.log | tail -2 | cut -c1-300 | tr '\n' ' ')"
fi
touch $P/logs/J_$J_JOB.done
