#!/bin/bash
# v21 P4: LongMemEval-Corr runs (+ LoCoMo-Corr) with one student.  $1 student (or "none"), $2 role.
#   main   comm21, comm21_sel (write-time sentence closure), comm21_overwrite (no version slots); LoCoMo + recurrence
#   nd|w|nv|nr|s1|s2   one ablation / seed student on LongMemEval-Corr (nd, nr also on LoCoMo)
#   zs     the 8B compressor as the same derivation-conditioned operator, zero-shot (no student)
# Profile records are precomputed (deriv_prof); the remaining closure units are annotated on demand by the same pass.
source j_common.sh
V=$1; ROLE=$2; RC=$P/results
serve_pair $V
CONC=16; [ "$FREE" -ge 12000 ] && CONC=48
INST=4; [ "$FREE" -ge 12000 ] && INST=6
run() { $PY run_corr.py --splits test200 --tag main3 --strategies "$1" --inst ${2:-$INST} --conc $CONC >> $P/logs/m3_${1%%,*}_v21.log 2>&1; }
need() { for s in "$@"; do [ "$(cnt $RC/main3/$s)" -eq 201 ] || fin "INCOMPLETE_$s=$(cnt $RC/main3/$s)"; done; }
case $ROLE in
  main)
    run comm21
    need comm21
    $PY locomo.py run --methods comm21 >> $P/logs/locomo_v21.log 2>&1
    $PY locomo.py run --regress --methods comm21 >> $P/logs/locomo_v21.log 2>&1
    [ "$(cnt $RC/locomo/comm21)" -ge 10 ] && [ "$(cnt $RC/locomo_reg/comm21)" -ge 10 ] || fin INCOMPLETE_LOCOMO
    run comm21_sel
    run comm21_overwrite
    need comm21_sel comm21_overwrite ;;
  nd|nr)
    S=comm21_$ROLE
    run $S
    need $S
    $PY locomo.py run --methods $S >> $P/logs/locomo_v21.log 2>&1
    [ "$(cnt $RC/locomo/$S)" -ge 10 ] || fin INCOMPLETE_LOCOMO_$S ;;
  w|nv|s1|s2)
    S=comm21_$ROLE
    run $S
    need $S ;;
  zs)
    run comm21_zs
    need comm21_zs ;;
  *) fin "UNKNOWN_ROLE_$ROLE" ;;
esac
log "eval $ROLE done"
touch $P/logs/J_$J_JOB.done
