#!/bin/bash
# Smoke test, phase A only (8B server with a small memory slice): derivation pass, relabel, data.  $1 GPU, $2 free MiB.
set -Eeuo pipefail
P=/data/yaokang/corrmem; cd $P/code; PY=../.venv/bin/python
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CM_EMB_DEVICE=cpu CM_CLE_DEVICE=cpu
G=${1:-3}; FREE=${2:-8000}; P8=18450
export CM_P8=$P8
L=$P/logs/smoke21.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
trap 'log "FAILED line=$LINENO status=$?"; bash stop_vllm.sh $P8' ERR
TOTAL=24564
F8=$(python3 -c "print(max(0.05, min(0.92, round(($FREE*0.88-600)/$TOTAL, 3))))")
CONC=8; [ "$FREE" -ge 14000 ] && CONC=32
rm -rf $P/models/smoke; mkdir -p $P/models/smoke/d21
log "smoke A start GPU=$G free=$FREE frac=$F8 conc=$CONC"
bash start_srv.sh $G $P8 /home/yaokang/models/Qwen3-8B-AWQ qwen3-8b $F8 >> $L
log "8B up"
$PY deriv.py annotate --kind lme --required-only --limit 2 --conc $CONC --inst 2 >> $L 2>&1
$PY deriv.py annotate --kind locomo --limit 1 --conc $CONC --inst 1 >> $L 2>&1
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
$PY commutator21.py relabel --src $P/models/w20 --out $P/models/smoke/d21 --limit 120 --conc $CONC >> $L 2>&1
$PY - <<'EOF' | tee -a $L
import json
E = [json.loads(l) for l in open('/data/yaokang/corrmem/models/smoke/d21/train.jsonl')]
print('pairs', len(E), 'pos', sum(e['y'] for e in E), 'active', sum(e['active'] for e in E))
for e in [e for e in E if e['y']][:5]:
    print('--', e['level'], e['old'][:90], '=>', e['new'][:90]); print('  D :', e['D'][:140]); print('  S :', e['sent'][:140])
    print('  D*:', e['target_rec'][:140]); print('  S*:', e['target'][:140], e.get('target_src'))
EOF
$PY commutator21.py data --src $P/models/smoke/d21 >> $L 2>&1
head -c 1200 $P/models/smoke/tnet21/train.jsonl | tee -a $L; echo
head -c 600 $P/models/smoke/tnet21w/train.jsonl | tee -a $L; echo
bash stop_vllm.sh $P8
log "SMOKE A OK"
