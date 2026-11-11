#!/bin/bash
# v21 P1: write-time derivation records (deriv.py), 8B compressor alone.
#   j_deriv.sh pairs I N    records of the units that occur in the training / held-out pairs (shard I of N);
#                           prompts of notes and digests are short, so a reduced window serves them
#   j_deriv.sh profiles     records of the profile units of every memory the experiments touch (LME training,
#                           held-out and test200 users; LoCoMo): their prompts list every claim, full window needed
# Units of the evaluation memories other than profiles are written on demand at correction time by the same pass.
source j_common.sh
MODE=$1
mkdir -p $P/states/deriv $P/states/deriv_locomo
PAIRS=$P/models/w20/train.jsonl,$P/models/w20/test.jsonl
case $MODE in
  pairs)
    I=$2; N=$3
    serve8
    CONC=24; [ "$FREE" -ge 12000 ] && CONC=64
    $PY deriv.py annotate --kind lme --required-only --units-from $PAIRS --shard $I:$N --conc $CONC --inst 4 >> $P/logs/q40_deriv_$I.log 2>&1 || fin DERIV_PAIRS_FAILED
    read -r NF ND NU NH < <($PY deriv.py count --kind lme --required-only --units-from $PAIRS --shard $I:$N 2>/dev/null || echo 0 0 0 0)
    # the shard is complete when every pair unit of its memories has a record, except units deferred to the full window
    DEF=$(grep -oE 'deferred_units [0-9]+' $P/logs/q40_deriv_$I.log | tail -1 | grep -oE '[0-9]+$' || echo 0)
    log "deriv pairs shard $I/$N: memories=$NF complete=$ND units=$NU with_record=$NH deferred=$DEF"
    [ "$((NH + DEF))" -ge "$NU" ] || fin "DERIV_PAIRS_INCOMPLETE $NH+$DEF/$NU" ;;
  profiles)
    serve8 24576
    $PY deriv.py annotate --kind lme --required-only --profiles-only --conc 8 --inst 4 >> $P/logs/q40_deriv_prof.log 2>&1 || fin DERIV_PROF_LME_FAILED
    $PY deriv.py annotate --kind locomo --profiles-only --conc 8 --inst 2 >> $P/logs/q40_deriv_prof.log 2>&1 || fin DERIV_PROF_LOCOMO_FAILED
    # the pair units deferred by the reduced-window shards (profiles and any other over-long unit)
    $PY deriv.py annotate --kind lme --required-only --units-from $PAIRS --conc 8 --inst 4 >> $P/logs/q40_deriv_prof.log 2>&1 || fin DERIV_PAIRS_FULL_FAILED
    read -r NF ND NU NH < <($PY deriv.py count --kind lme --required-only --profiles-only)
    read -r LF LD LU LH < <($PY deriv.py count --kind locomo --profiles-only)
    read -r PF PD PU PH < <($PY deriv.py count --kind lme --required-only --units-from $PAIRS)
    log "deriv profiles: lme $NH/$NU locomo $LH/$LU; pair units $PH/$PU"
    [ "$NH" -ge "$NU" ] && [ "$LH" -ge "$LU" ] && [ "$PH" -ge "$PU" ] || fin "DERIV_PROF_INCOMPLETE lme=$NH/$NU locomo=$LH/$LU pairs=$PH/$PU" ;;
  *) fin "UNKNOWN_MODE_$MODE" ;;
esac
touch $P/logs/J_$J_JOB.done
