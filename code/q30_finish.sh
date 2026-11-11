#!/bin/bash
set -Eeuo pipefail
P=/data/yaokang/corrmem
cd "$P/code"
PY=../.venv/bin/python
L=$P/logs/q30.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$L"; }
fin() { log "$1"; echo "$1" > "$P/logs/Q30.failed"; exit 1; }
trap 'fin "FAILED finish line=$LINENO status=$?"' ERR
exec 9>"$P/logs/q30.run.lock"
flock -n 9 || exit 1
rm -f "$P/logs/Q30.failed"
python3 $P/xtab.py > $P/results/xtab.txt 2>&1
X_LME=comm20,comm10,append,slots,simupdate,coarse,ncf,recompress,spec,rebuild X_LOC=comm20,comm10,ncf,coarse,recompress,rebuild,append,rawrag,membank,mem0,amem X_SEQ=comm20,comm10,ncf,coarse,recompress,rebuild X_OUT=extras30.json $PY extras6.py > $P/results/extras30.txt 2>&1
cp $P/models/w20/label_cost.json $P/results/label_cost_v20.json
for v in tnet20 tnet20nw tnet20nv tnet20nr; do cp $P/models/$v/train_meta.json $P/results/train_meta_$v.json; done
cd $P/results
shopt -s nullglob
tar czf $P/r30.tgz t30*.json stat30.json reg30.json xtab.json extras30.json suffbench_v20.json latency_v20.json witness_stats.json comm2_main3_v20.json coverage_main3_v20.json tnet_eval_tnet20*_v20.json judgecal_*.json label_cost_v20.json train_meta_tnet20*.json
cd $P/code
[ -s $P/results/t30_main_qwen25-7b.json ] && [ -s $P/results/suffbench_v20.json ] && [ -s $P/results/comm2_main3_v20.json ] || fin OUTPUT_MISSING
log "ALL DONE"
touch $P/logs/Q30.done
