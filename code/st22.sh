#!/bin/bash
# v22 status: scheduler log tail, job states, counts of finished result files, GPU free memory.
P=/data/yaokang/corrmem; R=$P/v22/results
echo "== sched22 $(pgrep -u $USER -f '[s]ched22.py' | tr '\n' ' ')  $(date '+%m-%d %H:%M')"
tail -n ${1:-8} $P/logs/sched22.log 2>/dev/null
echo "== jobs"
python3 - <<'EOF'
import json, time
try:
    st = json.load(open("/data/yaokang/corrmem/logs/sched22.state.json"))["jobs"]
except Exception as e:
    print("no state", e); st = {}
for k, s in st.items():
    if s.get("status") in ("running", "failed"):
        extra = f" gpus={s.get('gpus')} port={s.get('port')} {(time.time()-s['start'])/60:.0f}min" if s.get("status") == "running" else f" attempts={s.get('attempts')} retry_in={(s.get('next',0)-time.time())/60:.0f}min"
        print(f"  {k:22s} {s['status']}{extra}")
print("  done:", " ".join(k for k, s in st.items() if s.get("status") == "done"))
EOF
echo "== results"
for d in $R/main3/* $R/locomo/* $R/locomo_reg/*; do [ -d "$d" ] && echo -n "$(basename $(dirname $d))/$(basename $d)=$(ls $d | grep -c json) "; done; echo
echo "== states: lme $(ls $P/v22/states/cocomem 2>/dev/null | grep -c json)/274 locomo $(ls $P/v22/states/locomo_cocomem 2>/dev/null | grep -c json)/10  labels: $(ls $P/models/d22 2>/dev/null | tr '\n' ' ')"
echo "== gpu free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')  load: $(cut -d' ' -f1-3 /proc/loadavg)  mem avail: $(free -g | awk '/Mem/{print $7}')G"
