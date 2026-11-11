#!/bin/bash
# v22 P8 (CPU): per-question export of every system (published baselines from the v21 result root, CoCoMem v22 runs),
# analyze.py summaries, programmatic benchmark merge; packed into r22.tgz.
source j_common.sh
export CM_RUN_ROOT=$P/v22 CM_COMPRESSOR=plan
trap - EXIT
R0=$P/results; RC=$CM_RUN_ROOT/results
mkdir -p $RC/main3 $RC/locomo
for b in rawrag readagent amem ga membank mem0 oracle comm21 comm21_sel comm21_overwrite comm21_zs comm21_nd comm21_w comm21_nv comm21_nr; do
  [ -e $RC/main3/$b ] || [ ! -d $R0/main3/$b ] || ln -s $R0/main3/$b $RC/main3/$b
  [ -e $RC/locomo/$b ] || [ ! -d $R0/locomo/$b ] || ln -s $R0/locomo/$b $RC/locomo/$b
done
V22=comm22,comm22_t21,comm22_sel,comm22_overwrite,comm22_zs,comm22_nd,comm22_w,comm22_nv,comm22_nr,comm22_s1,comm22_s2,comm22_blind,recompress,rebuild,slots
ALL=rawrag,ga,membank,readagent,mem0,amem,comm21,comm21_sel,comm21_overwrite,comm21_zs,comm21_nd,comm21_w,comm21_nv,comm21_nr,comm21np,comm21np_sel,$V22
LOC=rawrag,ga,membank,readagent,mem0,amem,comm21,comm21_nd,comm21_nr,comm21np,comm22,comm22_t21,comm22_nd,comm22_nr,recompress,rebuild,slots
for k in qwen25-7b llama31; do
  $PY export_pub.py --key label_$k --methods $ALL --loc_methods $LOC --shared $V22 --canon recompress --out $RC/pub_export_$k.json > $RC/pub_export_$k.txt 2>&1 || true
done
# LoCoMo recurrence: every corrected question re-asked after each later correction (fresh / carried / recurrence)
$PY reg19.py comm22 > $RC/reg19.txt 2>&1 || true
# held-out pairs: does the record name the claim a verified correction changes (record-first vs attributed records)
$PY rec_recall22.py v22=$P/models/d22/test.jsonl v21=$P/models/d21/test.jsonl v22train=$P/models/d22/train.jsonl $RC/rec_recall_v22.json > $RC/rec_recall_v22.txt 2>&1 || true
# where a corrected answer is decided: intake, closure, record writes (per CoCoMem variant)
$PY err_quant22.py comm22,comm22_sel,comm22_blind,comm22_t21,comm21np,comm21np_sel,comm22_overwrite,comm22_zs,recompress,rebuild,slots,comm22_nd,comm22_w,comm22_nv,comm22_nr,comm22_s1,comm22_s2 > $RC/err_quant_v22.txt 2>&1 || true
# the records of the sentences a test correction rewrites (computation or restatement), per operator
$PY rewr_stats22.py comm22,comm22_sel,comm22_nd,comm22_w,comm22_nv,comm22_nr,comm22_s1,comm22_s2,recompress $RC/rewr_stats_v22.json > $RC/rewr_stats_v22.txt 2>&1 || true
$PY analyze.py --tag main3 --methods $ALL --ref comm22 --key label_qwen25-7b --json $RC/t22_main.json > $RC/t22_main.txt 2>&1 || true
$PY analyze.py --tag main3 --methods $ALL --ref comm22 --key label_qwen25-7b --all --json $RC/t22all_main.json > $RC/t22all_main.txt 2>&1 || true
$PY analyze.py --tag locomo --methods $LOC --ref comm22 --key label_qwen25-7b --json $RC/t22_locomo.json > $RC/t22_locomo.txt 2>&1 || true
$PY - <<'EOF'
import glob, json, os
R = os.environ["CM_RUN_ROOT"] + "/results"
out = {}
for role in ("main", "nd", "w", "nv", "nr", "ref"):
    f = f"{R}/suffbench_v21_{role}.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    for k in ("n", "n_all", "labels", "record"):
        out.setdefault(k, d.get(k))
    for k in ("acc", "tokens", "chain"):
        dst = out.setdefault(k, {})
        for kk, vv in (d.get(k) or {}).items():
            if kk in ("record_ok", "operands_ok", "step", "step_all", "theorem", "recover", "step_kind") and isinstance(vv, dict):
                dst.setdefault(kk, {}).update(vv)               # per-system measurements along the sequences
            elif kk in ("n", "length", "step_n"):
                dst.setdefault(kk, vv)                          # the sequences are the same in every role (main first)
            else:
                dst[kk] = vv
json.dump(out, open(f"{R}/suffbench_v22.json", "w"), indent=1)
print("suffbench merged:", sorted((out.get("acc") or {}).keys()))
EOF
$PY rebuild_units.py stats > $P/logs/v22_write_stats.log 2>&1 || true
for f in $R0/comm2_main3_comm21*.json; do [ -e $RC/$(basename $f) ] || cp $f $RC/; done   # v21 rows next to v22 rows
python3 $P/xtab.py --results $RC > $RC/xtab.txt 2>&1 || true          # Sound / Stale per method from comm2_main3_<m>.json
cp $P/models/d22/label_cost.json $RC/label_cost_v22.json 2>/dev/null || true
for v in tnet22 tnet22nd tnet22w tnet22nv tnet22nr tnet22_s1 tnet22_s2; do cp $P/models/$v/train_meta.json $RC/train_meta_$v.json 2>/dev/null || true; done
cd $RC
shopt -s nullglob
cp $P/results/latency_v22.json $P/results/latency_v22_env.json $RC/ 2>/dev/null || true
tar czf $P/r22.tgz.tmp pub_export_*.json pub_export_*.txt t22*.json t22*.txt suffbench_v22.json suffbench_v21_[mnwr]*.json write_stats_v22.json label_cost_v22.json train_meta_tnet22*.json tnet_eval_tnet22*.json latency_v22*.json xtab.json xtab.txt comm2_main3_v22*.json err_quant_v22.* reg19.* rec_recall_v22.json rewr_stats_v22.*
mv $P/r22.tgz.tmp $P/r22.tgz
cd $P/code
[ -s $RC/pub_export_qwen25-7b.json ] || fin OUTPUT_MISSING
log "tables done: $(ls -la $P/r22.tgz | cut -c1-80)"
touch $P/logs/J_$J_JOB.done
