#!/bin/bash
# v22 P6: programmatic benchmark (suffbench5, with per-step record correctness) with one student.
# $1 student (or none), $2 role (main|nd|w|nv|nr|ref), $3 "cached": the units of the benchmark and the 8B rows are
# already in the response cache (written by an earlier run of role main), so only the student is served.
source j_common.sh
export CM_RUN_ROOT=$P/v22 CM_COMPRESSOR=plan
V=$1; ROLE=$2
case $ROLE in
  main) SYS=copy,comm22,comm22_raw,ledger_all,teacher_d ;;
  nd) SYS=comm22_nd ;;  w) SYS=comm22_w ;;  nv) SYS=comm22_nv ;;  nr) SYS=comm22_nr ;;
  ref) SYS=think_all ;;
  *) fin "UNKNOWN_ROLE_$ROLE" ;;
esac
mkdir -p $CM_RUN_ROOT/results
if [ ! -s $CM_RUN_ROOT/results/suffbench_v21_$ROLE.json ]; then
  if [ "$V" = none ]; then serve8; ST=""
  elif [ "${3:-}" = cached ]; then serve_student $V; ST="--students $V:$P4"
  else serve_pair $V; ST="--students $V:$P4"; fi
  $PY suffbench5.py $ST --systems $SYS --tag $ROLE >> $P/logs/v22_suff_$ROLE.log 2>&1 || fin SUFFBENCH_FAILED
fi
[ -s $CM_RUN_ROOT/results/suffbench_v21_$ROLE.json ] || fin SUFF_OUTPUT_MISSING
log "suffbench $ROLE done"
touch $P/logs/J_$J_JOB.done
