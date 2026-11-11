#!/bin/bash
# v21 end-to-end latency without the LLM cache: every strategy on the same servers, one correction at a time.
# The compressor keeps its full 24k window here because the LLM repairs (Unit edit, Recompress, Rebuild) read whole
# profiles; the student (comm21 only) is served on the second GPU of the placement, or on the same GPU when it fits.
source j_common.sh
if [ -s $P/results/latency_v21.json ]; then touch $P/logs/J_$J_JOB.done; exit 0; fi
serve_pair tnet21 24576
CM_NOCACHE=1 $PY bench_latency.py --n 30 --methods comm21,comm21_sel,ncf,coarse,recompress,rebuild --out latency_v21.json > $P/logs/q40_lat.log 2>&1 || fin LATENCY_FAILED
[ -s $P/results/latency_v21.json ] || fin LATENCY_MISSING
echo "{\"cle_device\": \"$CM_CLE_DEVICE\", \"emb_device\": \"$CM_EMB_DEVICE\", \"gpu8\": \"$G1\", \"gpu4\": \"$G2\", \"free\": $FREE, \"free2\": $FREE2, \"window\": $CM_MAXLEN}" > $P/results/latency_v21_env.json
log "latency done"
touch $P/logs/J_$J_JOB.done
