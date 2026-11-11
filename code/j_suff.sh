#!/bin/bash
# v21 P5: programmatic benchmark (suffbench5: program-computed truth, 100-step chains with retractions) with one
# student.  $1 student (or none), $2 role (main|nd|w|nv|nr|ref).  The 8B serves the reference systems (ledger_all,
# teacher_d, think_all) and the write-time record pass of the synthetic units.
source j_common.sh
V=$1; ROLE=$2
case $ROLE in
  main) SYS=copy,comm21,ledger_all,teacher_d ;;
  nd) SYS=comm21_nd ;;  w) SYS=comm21_w ;;  nv) SYS=comm21_nv ;;  nr) SYS=comm21_nr ;;
  ref) SYS=think_all ;;
  *) fin "UNKNOWN_ROLE_$ROLE" ;;
esac
if [ ! -s $P/results/suffbench_v21_$ROLE.json ]; then
  if [ "$V" = none ]; then serve8; ST=""; else serve_pair $V; ST="--students $V:$P4"; fi
  $PY suffbench5.py $ST --systems $SYS --tag $ROLE >> $P/logs/q40_suff_$ROLE.log 2>&1 || fin SUFFBENCH_FAILED
fi
[ -s $P/results/suffbench_v21_$ROLE.json ] || fin SUFF_OUTPUT_MISSING
log "suffbench $ROLE done"
touch $P/logs/J_$J_JOB.done
