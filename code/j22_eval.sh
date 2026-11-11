#!/bin/bash
# v22 P2/P5: LongMemEval-Corr runs (+ LoCoMo-Corr) on the record-first memories.  $1 student (or "none"), $2 role.
#   main    comm22, comm22_sel (write-time sentence closure), comm22_overwrite (no version slots); LoCoMo + recurrence
#   t21     the v21 student on the record-first memories (comm22_t21), LME + LoCoMo
#   recomp  recompute references on the same memories: recompress (closure, reads the ledger) and rebuild (everything);
#           slots = the ledger alone is maintained, the derived units are left as they are
#   zs      the 8B compressor as the same derivation-conditioned operator, zero-shot
#   nd|nr   ablation students on LME + LoCoMo;  w|nv|s1|s2  on LME
#   blind   comm22 with the claims of an arriving session extracted from the session alone (comm22_blind)
#   v21     the v21 memories without their profile + the v21 student (comm21np), LME + LoCoMo
source j_common.sh
export CM_RUN_ROOT=$P/v22 CM_COMPRESSOR=plan
V=$1; ROLE=$2; RC=$CM_RUN_ROOT/results
case $ROLE in
  recomp) serve_pair none 24576 ;;
  *) serve_pair $V ;;
esac
CONC=16; [ "$FREE" -ge 12000 ] && CONC=48
INST=4; [ "$FREE" -ge 12000 ] && INST=6
run() { $PY run_corr.py --splits test200 --tag main3 --strategies "$1" --inst ${2:-$INST} --conc $CONC >> $P/logs/v22_m3_${1%%,*}.log 2>&1; }
need() { for s in "$@"; do [ "$(cnt $RC/main3/$s)" -eq 201 ] || fin "INCOMPLETE_$s=$(cnt $RC/main3/$s)"; done; }
loco() { for s in "$@"; do $PY locomo.py run --methods $s >> $P/logs/v22_locomo.log 2>&1; [ "$(cnt $RC/locomo/$s)" -ge 10 ] || fin "INCOMPLETE_LOCOMO_$s"; done; }
case $ROLE in
  main)
    run comm22; need comm22
    loco comm22
    $PY locomo.py run --regress --methods comm22 >> $P/logs/v22_locomo.log 2>&1
    [ "$(cnt $RC/locomo_reg/comm22)" -ge 10 ] || fin INCOMPLETE_LOCOMO_REG
    run comm22_sel; run comm22_overwrite; need comm22_sel comm22_overwrite ;;
  t21)
    run comm22_t21; need comm22_t21
    loco comm22_t21 ;;
  recomp)
    run recompress 2; run rebuild 2; run slots; need recompress rebuild slots
    loco recompress rebuild slots ;;
  zs)
    run comm22_zs; need comm22_zs ;;
  nd|nr)
    S=comm22_$ROLE
    run $S; need $S
    loco $S ;;
  w|nv|s1|s2|blind)
    S=comm22_$ROLE
    run $S; need $S ;;
  v21)
    # write-path control: the prose-first memories with attributed records and their own student, on the same two
    # derived levels (strip_profile.py) and through the same intake; results go to the v22 result root (comm21np)
    export CM_RUN_ROOT=$P/v21np CM_COMPRESSOR=text
    $PY strip_profile.py >> $P/logs/v22_m3_comm21np.log 2>&1 || fin STRIP_FAILED
    run comm21np; need comm21np
    loco comm21np ;;
  v21sel)
    # the sentence closure S(c) of the control: the same memories, intake and student as comm21np, the operator applied
    # only to the sentences whose attributed record or citations name the corrected claim
    export CM_RUN_ROOT=$P/v21np CM_COMPRESSOR=text
    run comm21np_sel; need comm21np_sel ;;
  *) fin "UNKNOWN_ROLE_$ROLE" ;;
esac
log "eval $ROLE done"
touch $P/logs/J_$J_JOB.done
