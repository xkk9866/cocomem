#!/bin/bash
# v22 end-to-end latency without the LLM cache on the record-first memories: CoCoMem, CoCoMem-S(c) and the two
# recompute references (Recompress closure, Rebuild all) repair the same 30 corrections one at a time on the same
# servers.  The compressor keeps its full 24k window (the recompute references read whole profiles).
source j_common.sh
export CM_RUN_ROOT=$P/v22 CM_COMPRESSOR=plan
if [ -s $P/results/latency_v22.json ]; then cp $P/results/latency_v22.json $CM_RUN_ROOT/results/; touch $P/logs/J_$J_JOB.done; exit 0; fi
serve_pair tnet22 24576
CM_NOCACHE=1 $PY bench_latency.py --n 30 --methods comm22,comm22_sel,recompress,rebuild --out latency_v22.json > $P/logs/v22_latency.log 2>&1 || fin LATENCY_FAILED
[ -s $P/results/latency_v22.json ] || fin LATENCY_MISSING
echo "{\"gpu8\": \"$G1\", \"gpu4\": \"$G2\", \"free\": $FREE, \"free2\": $FREE2, \"window\": $CM_MAXLEN}" > $P/results/latency_v22_env.json
cp $P/results/latency_v22.json $P/results/latency_v22_env.json $CM_RUN_ROOT/results/
log "latency done"
touch $P/logs/J_$J_JOB.done
