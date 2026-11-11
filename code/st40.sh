#!/bin/bash
# compact status of the v21 queue
P=/data/yaokang/corrmem
date '+%m-%d %H:%M:%S'
echo "--- gpu (idx used MiB)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' '; echo
echo "--- sched (last 6)"; tail -n 6 $P/logs/sched21.log
echo "--- running"; python3 - <<'EOF'
import json, time
try:
    st = json.load(open('/data/yaokang/corrmem/logs/sched21.state.json'))
except Exception as e:
    print('no state', e); st = {'jobs': {}}
for j, s in st['jobs'].items():
    if s.get('status') == 'running':
        print(f"  {j}: gpus={s.get('gpus')} port={s.get('port')} {(time.time()-s['start'])/60:.0f} min")
    elif s.get('status') == 'failed':
        print(f"  {j}: FAILED x{s.get('attempts')} retry in {(s.get('next',0)-time.time())/60:.0f} min")
EOF
echo "--- done markers"; ls $P/logs/J_*.done 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '; echo
echo "--- counts"; echo "deriv_files=$(ls $P/states/deriv 2>/dev/null | wc -l) deriv_loc=$(ls $P/states/deriv_locomo 2>/dev/null | wc -l) d21=$(ls $P/models/d21 2>/dev/null | tr '\n' ' ')"
echo "pair units (mem done sel have): $(cd $P/code && ../.venv/bin/python deriv.py count --kind lme --required-only --units-from $P/models/w20/train.jsonl,$P/models/w20/test.jsonl 2>/dev/null)  profiles: $(cd $P/code && ../.venv/bin/python deriv.py count --kind lme --required-only --profiles-only 2>/dev/null)"
echo "models: $(for v in tnet21 tnet21nd tnet21nv tnet21nr tnet21w tnet21_s1 tnet21_s2; do [ -e $P/models/$v/merged/config.json ] && echo -n "$v "; done)"
for s in comm21 comm21_sel comm21_overwrite comm21_zs comm21_nd comm21_w comm21_nv comm21_nr comm21_s1 comm21_s2; do echo -n "$s=$(ls $P/results/main3/$s 2>/dev/null | wc -l) "; done; echo
echo "loc: $(for s in comm21 comm21_nd comm21_nr; do echo -n "$s=$(ls $P/results/locomo/$s 2>/dev/null | wc -l) "; done) reg=$(ls $P/results/locomo_reg/comm21 2>/dev/null | wc -l)"
echo "--- job logs (last line each)"; for f in $P/logs/j_*.log; do echo "$(basename $f): $(tail -n 1 $f | cut -c1-150)"; done 2>/dev/null
