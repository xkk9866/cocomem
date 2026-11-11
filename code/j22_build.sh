#!/bin/bash
# v22 P1: record-first rebuild of the derived units (rebuild_units.py), 8B compressor alone.  Every pass merges the
# units it has written into the memory's state file; a memory is complete when no unit is pending.
#   j22_build.sh pairs     the units of the training / held-out pairs; the labelling starts when they are written
#   j22_build.sh lme I N   shard I of N: the other units of the LME memories (test200 + pair users)
#   j22_build.sh full      whatever is still pending, the LoCoMo memories, completeness check, write statistics
source j_common.sh
export CM_RUN_ROOT=$P/v22 CM_COMPRESSOR=plan
MODE=$1
mkdir -p $CM_RUN_ROOT/states $CM_RUN_ROOT/results
[ -e $CM_RUN_ROOT/models ] || ln -s ../models $CM_RUN_ROOT/models
[ -e $CM_RUN_ROOT/data ] || ln -s ../data $CM_RUN_ROOT/data
count() { read -r NM ND NP NU NPP < <($PY rebuild_units.py count --kind lme); }
case $MODE in
  pairs)
    serve8
    $PY rebuild_units.py run --kind lme --units pairs --conc 64 --inst 8 >> $P/logs/v22_build_pairs.log 2>&1 || fin BUILD_FAILED
    $PY rebuild_units.py stats --partial > $P/logs/v22_write_stats_pairs.log 2>&1 || log "partial write stats failed (non-fatal)"
    count; log "build pairs done: memories=$NM complete=$ND pending units=$NU (pair units among them: $NPP)" ;;
  lme)
    I=$2; N=$3
    serve8
    $PY rebuild_units.py run --kind lme --shard $I:$N --units rest --conc 64 --inst 6 >> $P/logs/v22_build_$I.log 2>&1 || fin BUILD_FAILED
    count; log "build lme shard $I/$N done: memories=$NM complete=$ND pending units=$NU" ;;
  full)
    serve8 24576
    $PY rebuild_units.py strip --kind lme >> $P/logs/v22_build_full.log 2>&1 || fin STRIP_FAILED
    $PY rebuild_units.py run --kind lme --pending-only --conc 64 --inst 6 >> $P/logs/v22_build_full.log 2>&1 || fin BUILD_PENDING_FAILED
    $PY rebuild_units.py run --kind locomo --conc 64 --inst 4 >> $P/logs/v22_build_full.log 2>&1 || fin BUILD_LOCOMO_FAILED
    count
    read -r LM LD LP LU LPP < <($PY rebuild_units.py count --kind locomo)
    log "build full: lme $ND/$NM pending=$NP; locomo $LD/$LM pending=$LP"
    [ "$ND" -eq "$NM" ] && [ "$LD" -eq "$LM" ] || fin "BUILD_INCOMPLETE lme=$ND/$NM locomo=$LD/$LM"
    $PY rebuild_units.py stats > $P/logs/v22_write_stats.log 2>&1 || log "write stats failed (non-fatal)" ;;
  *) fin "UNKNOWN_MODE_$MODE" ;;
esac
touch $P/logs/J_$J_JOB.done
