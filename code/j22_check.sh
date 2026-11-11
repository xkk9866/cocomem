#!/bin/bash
# v22 P0: the intake of correction sessions on memories outside the test split (resolve_check.py): the two-step
# intake next to the ledger-conditioned resolution.  One 8B server of its own, so that no other job shares its cache.
#   j22_check.sh [limit]
source j_common.sh
unset CM_COMPRESSOR
serve8
CONC=8; [ "$FREE" -ge 12000 ] && CONC=16
$PY resolve_check.py $P8 ${1:-0} $CONC > $P/logs/resolve_check.out 2>&1 || fin RESOLVE_CHECK_FAILED
log "resolve check done: $(grep -c RESOLVE $P/logs/resolve_check.out) revised slots listed"
touch $P/logs/J_$J_JOB.done
