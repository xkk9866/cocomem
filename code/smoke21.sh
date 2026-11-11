#!/bin/bash
# End-to-end smoke test of the v21 pipeline on tiny inputs, on ONE GPU ($1), mirroring the scheduler's job phases:
# A) 8B alone: derivation pass, relabel, data;  B) training (GPU only);  C) 8B + student: eval, runs, benchmark, latency.
set -Eeuo pipefail
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu
G=${1:-3}; P8=18450; P4=18451
export CM_P8=$P8 CM_PT=$P4 J_GPU=$G J_PORT=$P8 J_FREE=${2:-19000} J_JOB=smoke
L=$P/logs/smoke21.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
trap 'log "FAILED line=$LINENO status=$?"; bash stop_vllm.sh $P8; bash stop_vllm.sh $P4' ERR
TOTAL=24564
frac() { python3 -c "import sys; print(max(0.05, min(0.92, round(float(sys.argv[1])/$TOTAL, 3))))" "$1"; }
FREE=$J_FREE
F8_ALONE=$(frac $((FREE*88/100 - 600))); F8_PAIR=$(frac $((FREE*55/100 - 500))); F4_PAIR=$(frac $((FREE*37/100 - 400)))
rm -rf $P/models/smoke $P/results/smoke; mkdir -p $P/models/smoke/d21
log "smoke start GPU=$G free=$FREE fracs $F8_ALONE $F8_PAIR $F4_PAIR"
# ---------------------------------------------------------------- A
bash start_srv.sh $G $P8 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b $F8_ALONE >> $L
log "8B up"
$PY deriv.py annotate --kind lme --required-only --limit 2 --conc 32 >> $L 2>&1
$PY deriv.py annotate --kind locomo --limit 1 --conc 32 >> $L 2>&1
log "deriv: $(ls $P/states/deriv | wc -l) lme, $(ls $P/states/deriv_locomo | wc -l) locomo"
$PY - <<'EOF' | tee -a $L
import json, glob
f = sorted(glob.glob('/data/yaokang/corrmem/states/deriv/*.json'))[0]
d = json.load(open(f))
print('usage', d['usage'])
for uid, v in list(d['units'].items())[:3] + [(k, v) for k, v in d['units'].items() if k.startswith('profile')][:1]:
    print('==', uid, len(v['sents']), 'sentences')
    for s, r, i in list(zip(v['sents'], v['deriv'], v['ids']))[:6]:
        print('  S:', s[:110]); print('  D:', r[:140]); print('  ids:', i)
EOF
$PY commutator21.py relabel --src $P/models/w20 --out $P/models/smoke/d21 --limit 150 --conc 64 >> $L 2>&1
$PY - <<'EOF' | tee -a $L
import json
E = [json.loads(l) for l in open('/data/yaokang/corrmem/models/smoke/d21/train.jsonl')]
print('pairs', len(E), 'pos', sum(e['y'] for e in E), 'active', sum(e['active'] for e in E))
for e in [e for e in E if e['y']][:4]:
    print('--', e['level'], e['old'][:90], '=>', e['new'][:90]); print('  D :', e['D'][:140]); print('  S :', e['sent'][:140])
    print('  D*:', e['target_rec'][:140]); print('  S*:', e['target'][:140], e.get('target_src'))
EOF
$PY commutator21.py data --src $P/models/smoke/d21 >> $L 2>&1
head -c 1500 $P/models/smoke/tnet21/train.jsonl | tee -a $L; echo
bash stop_vllm.sh $P8
# ---------------------------------------------------------------- B
log "training tiny student on GPU $G"
CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY commutator21.py train --out $P/models/smoke/tnet21 --max_len 2048 --tok_budget 2048 >> $L 2>&1
log "trained: $(head -c 300 $P/models/smoke/tnet21/train_meta.json)"
# ---------------------------------------------------------------- C
bash start_srv.sh $G $P8 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b $F8_PAIR >> $L
bash start_tnet.sh $P/models/smoke/tnet21/merged $F4_PAIR tnet21 $P4 $G >> $L
log "8B + student up"
$PY commutator21.py eval --out $P/models/smoke/tnet21 --served tnet21 --port $P4 --suffix _smoke >> $L 2>&1
$PY run_corr.py --splits test200 --tag smoke --strategies comm21,comm21_sel,comm21_zs --limit 2 --inst 2 --conc 32 >> $L 2>&1
$PY - <<'EOF' | tee -a $L
import json, glob
for f in sorted(glob.glob('/data/yaokang/corrmem/results/smoke/*/*.json')):
    d = json.load(open(f)); r = d['repair'].get('S1', {})
    print(f.split('/')[-2], d['qid'], 'closure', r.get('closure'), 'mapped', r.get('mapped_sents'), '/', r.get('closure_sents'),
          'rewritten', r.get('rewritten_sents'), 't_usage', r.get('t_usage'), 'usage', r.get('usage'))
EOF
$PY suffbench5.py --units 3 --chain_units 1 --chain_len 3 --students tnet21:$P4 --systems copy,comm21,teacher_d,ledger_all --tag smoke >> $L 2>&1
tail -c 1500 $P/results/suffbench_v21_smoke.json | tee -a $L; echo
CM_NOCACHE=1 $PY bench_latency.py --n 2 --methods comm21,comm21_sel --out latency_smoke.json >> $L 2>&1
cat $P/results/latency_smoke.json | tee -a $L
bash stop_vllm.sh $P8; bash stop_vllm.sh $P4
rm -rf $P/results/smoke $P/results/latency_smoke.json $P/results/suffbench_v21_smoke.json $P/results/suffbench_v21_cases_smoke.json $P/results/tnet_eval_tnet21_smoke.json
log "SMOKE OK"
