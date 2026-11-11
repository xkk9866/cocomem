#!/bin/bash
# v21 P7: tables and statistics (CPU only) -> results/t40_*.json, stat40.json, reg40.json, xtab.json, extras40.json,
# suffbench_v21.json, deriv_stats.json; packed into r40.tgz.
source j_common.sh
trap - EXIT
RC=$P/results
$PY - <<'EOF'
import glob, json, os
R = os.environ.get("CM_ROOT", "/data/yaokang/corrmem") + "/results"
out = {}
for f in sorted(glob.glob(f"{R}/suffbench_v21_*.json")):
    if "_cases_" in f:
        continue
    if os.path.basename(f) not in {f"suffbench_v21_{role}.json" for role in ("main","nd","w","nv","nr","ref")}:
        continue
    d = json.load(open(f))
    for k in ("n", "labels", "record"):
        if k in ("n", "labels") and k in out and out[k] != d[k]:
            raise ValueError(f"Different programmatic cases in {f}")
        out.setdefault(k, d.get(k))
    for k in ("acc", "tokens", "chain"):
        out.setdefault(k, {}).update({kk: vv for kk, vv in (d.get(k) or {}).items()})
json.dump(out, open(f"{R}/suffbench_v21.json", "w"), indent=1)
print("suffbench merged:", sorted((out.get("acc") or {}).keys()))
EOF
M=append,slots,simupdate,coarse,ncf,recompress,spec,rebuild,comm10,comm20,comm21,comm21_sel,comm21_overwrite,comm21_zs,comm21_nd,comm21_w,comm21_nv,comm21_nr,comm21_s1,comm21_s2,rawrag,membank,mem0,oracle
for k in qwen25-7b; do
  $PY analyze.py --tag main3 --methods $M --ref comm21 --key label_$k --json $RC/t40_main_$k.json > $RC/t40_main_$k.txt 2>&1
done
$PY analyze.py --tag main3 --methods $M --ref comm21 --key label_qwen25-7b --all --json $RC/t40all_main_qwen25-7b.json > $RC/t40all_main_qwen25-7b.txt 2>&1
for b in off_mem0 off_amem; do
  Q=$(ls $RC/main3/$b | grep 'json$' | sed 's/.json$//' | paste -sd,)
  $PY analyze.py --tag main3 --methods rawrag,membank,$b,ncf,rebuild,comm20,comm21,oracle --ref comm21 --key label_qwen25-7b --qids "$Q" --json $RC/t40${b#off_}_main_qwen25-7b.json > $RC/t40${b#off_}_main_qwen25-7b.txt 2>&1
done
LO=append,slots,simupdate,coarse,ncf,recompress,rebuild,comm10,comm20,comm21,comm21_nd,comm21_nr,rawrag,membank,mem0
for k in qwen25-7b; do
  $PY analyze.py --tag locomo --methods $LO --ref comm21 --key label_$k --json $RC/t40_locomo_$k.json > $RC/t40_locomo_$k.txt 2>&1
done
$PY stat19.py comm21 append,simupdate,coarse,ncf,recompress,rebuild,comm10,comm20,comm21_sel,comm21_overwrite,comm21_zs,comm21_nd,comm21_w,comm21_nv,comm21_nr,comm21_s1,comm21_s2,rawrag,membank,mem0 > $RC/stat40.txt 2>&1
cp $RC/stat19.json $RC/stat40.json
$PY reg19.py comm21,comm20,ncf,coarse,rebuild,append > $RC/reg40.txt 2>&1
cp $RC/reg19.json $RC/reg40.json
python3 $P/xtab.py > $RC/xtab.txt 2>&1
X_LME=comm21,comm21_sel,comm21_overwrite,comm21_zs,comm20,comm10,append,slots,simupdate,coarse,ncf,recompress,spec,rebuild X_LOC=comm21,comm21_nd,comm21_nr,comm20,comm10,ncf,coarse,recompress,rebuild,append,rawrag,membank,mem0,amem X_SEQ=comm21,comm21_nd,comm21_nr,comm20,ncf,coarse,recompress,rebuild X_OUT=extras40.json $PY extras6.py > $RC/extras40.txt 2>&1
$PY deriv.py stats --required-only --pairs $P/models/tnet21/test.jsonl --out deriv_stats.json > $P/logs/q40_dstats.log 2>&1
cp $P/models/d21/label_cost.json $RC/label_cost_v21.json 2>/dev/null
for v in tnet21 tnet21nd tnet21w tnet21nv tnet21nr tnet21_s1 tnet21_s2; do cp $P/models/$v/train_meta.json $RC/train_meta_$v.json 2>/dev/null; done
$PY validate_core_v21.py "$RC"
cd $RC
shopt -s nullglob
tar czf $P/r40.tgz.tmp t40*.json t40*.txt stat40.json stat40.txt reg40.json reg40.txt xtab.json extras40.json extras40.txt suffbench_v21.json suffbench_v21_*.json latency_v21.json latency_v21_env.json deriv_stats.json comm2_main3_v21.json comm2_main3_comm21*.json label_cost_v21.json train_meta_tnet21*.json comm2_main3_v20.json xtab.txt tnet_eval_tnet21*.json
mv $P/r40.tgz.tmp $P/r40.tgz
cd $P/code
[ -s $RC/t40_main_qwen25-7b.json ] && [ -s $RC/suffbench_v21.json ] || fin OUTPUT_MISSING
log "tables done: $(ls -la $P/r40.tgz | cut -c1-80)"
touch $P/logs/J_$J_JOB.done
